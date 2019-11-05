import numpy as np
import matplotlib.pyplot as plt
from pyopencl.elementwise import ElementwiseKernel as eltWise
from pyopencl.reduction import ReductionKernel
from scipy.spatial.distance import cdist
from sklearn import datasets
import pyopencl as cl
import pyopencl.array as cl_array

deviceID = 0
platformID = 0
N = 10

dev = cl.get_platforms()[platformID].get_devices()[deviceID]

ctx = cl.Context([dev])
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
a_np = np.random.randn(N).astype(np.float32)
a_g = cl.array.to_device(queue, a_np)
# @profile
def error(D, d):
    N, _ = D.shape
    c = 0
    E = 0
    for j in range(0, N):
        for i in range(0, N):
            if i != j:
                E += ((D[i, j] - d[i, j]) ** 2) / D[i, j]

    return E.astype(np.float32)


def gradient(D, d, y):
    N, M = y.shape
    g = np.zeros([N, M])

    for q in range(0, M):
        for p in range(0, N):
            s = 0
            for j in range(0, N):
                if p != j:
                    s += (D[p, j] - d[p, j]) / (D[p, j] * d[p, j]) * (y[p, q] - y[j, q])
            g[p, q] = -s
    return g

def cl_gradient(D,d,y):
    N,M=y.shape
    g = np.zeros([N,M])
    res = np.empty([2201],dtype=float)
    # kernel_code ="""
    # __kernel float32 matrice(float32 D, float32 d,float32 N,float32 M,float32 g){
    #
    #  for(int q=0;q<M;q++){
    #
    #     for(int p=0;p<N;p++{
    #         float32 temp =0;
    #             for(int j=0;j<N;j++){
    #                 temp += (D[p][j]-d[p][j] / D[p][j] * d[p][j]);
    #             }
    #         res[p][q]=-s ;
    #      }
    # }
    # return res ;
    #
    # }
    # """
    matrice = eltWise(ctx,'float2 D, float2 d ,const float N,const float M, const float g,float res ',
                      '''
                        
                      for(int q=0;q<M;q++){
                        
                        for(int p=0;p<N;p++){
                            float temp =0;
                            for(int j=0;j<N;j++){
                                temp += (D[p][j]-d[p][j] / D[p][j] * d[p][j]);
                            }
                            res[p][q]=-temp ; 
                        }
                      }
                      ''','matrice')
    #program = cl.Program(ctx,matrice).build
    #program.matrice(queue,None,D,d,N,M,g)
    matrice(D,d,N,M,g,res)

    
def hessian(D, d, y):
    """Calcul de la dérivée seconde """
    N, M = y.shape
    h = np.zeros([N, M])

    for q in range(0, M):
        for p in range(0, N):
            s = 0
            for j in range(0, N):
                if p != j:
                    s += (
                            1
                            / (D[p, j] * d[p, j])
                            * (
                                    (D[p, j] - d[p, j])
                                    - (y[p, q] - y[j, q]) ** 2
                                    / d[p, j]
                                    * (1 + (D[p, j] - d[p, j]) / d[p, j])
                            )
                    )
            h[p, q] = -s
    return h


def y_update_constant(y, s, D, d, alpha=0.3):
    y = y + alpha * s
    d = cdist(y, y).astype(np.float32)
    E_new = cl_error(D, d)
    return y, E_new

#@profile
def cl_error(D, d):
    cl_D= cl.array.to_device(queue, D)
    cl_d= cl.array.to_device(queue, d)
    krnl = ReductionKernel(ctx, np.float32, neutral="0",
                       reduce_expr="a+b", map_expr="x[i] != 0 ? ((x[i] - y[i])*(x[i] - y[i]))/x[i]: 0.0",
                       arguments="__global float *x, __global float *y")
    res_error = krnl(cl_D,cl_d).get()
    return res_error


def y_update_halving(y, s, D, d, maxhalves=20):
    #E = error(D, d)
    E=cl_error(D,d)
    y_init = y
    for j in range(maxhalves):
        y = y_init + s
        d = cdist(y, y).astype(np.float32)
        E_new = cl_error(D, d)

        if E_new < E:
            break
        else:
            s = 0.5 * s
    return y, E_new


def acp(x, n):
    """ calcule les projections sur les n premieres composantes principales de x """
    [UU, DD, _] = np.linalg.svd(x)
    return UU[:, :n] * DD[:n]


#@profile
def main():
    maxhalves = 20  # nombre 1/2 pas maximum (step halving)
    alpha = 0.3  # constante pour la descente \alpha(t)=cste
    maxiter = 500  # nombre maximum d'itérations
    prec = 1e-9  # seuil du l'évolution de l'erreur
    n = 2  # dimension de l'espace de représentation : 2 ou 3

    # Chargement des données iris
    iris = datasets.load_iris()
    # ou en utilsant le fichier "iris.pickle" sur Moodle :
    # import pickle
    # with open("iris.pickle", 'rb') as f:
    #   iris = pickle.load(f)
    (x, index) = np.unique(iris.data, axis=0, return_index=True)
    target = iris.target[index]
    names = iris.target_names

    # Données dans l'espace initial
    N, M = x.shape  # N vecteurs de taille M

    # calcul des distances entre vecteurs de l'espace initial
    D = cdist(x, x).astype(np.float32)

    # calcul du facteur d'échelle
    scale = 1 / D.sum()

    # initalisations des vecteurs dans l'espace de représentation
    y = acp(x, n)

    # calcul des distances entre vecteurs de l'espace de représentation
    d = cdist(y, y).astype(np.float32)

    # calcul de l'erreur initiale
    E = cl_error(D, d)
    print(E.dtype)

    # Un maximum de maxiter itérations pour converger
    for i in range(maxiter):
        # Calcul du gradient
        g = gradient(D, d, y)
        cl_g= cl_gradient(D,d,y)

        # et de la Hessienne
        H = hessian(D, d, y)

        # calcul du pas
        s = -g.flatten(order="F") / np.abs(H.flatten(order="F"))
        s = np.reshape(s, (-1, n), order='F')
        # mise à jour des distance dans l'espace de représentation
        d = cdist(y, y).astype(np.float32)

        # mise à jour de y et de l'erreur
        y, E_new = y_update_halving(y, s, D, d, maxhalves=maxhalves)
        # y, E_new = y_update_constant(y, s, D, d, alpha=alpha)

        # si la variation de l'erreur est suffisament faible : arrêt
        if abs((E - E_new) / E) < prec:
            print("Précision atteinte: Optimisation terminée")
            break

        E = E_new

        print("epoch = %d : E = %12.10f" % (i + 1, E * scale))

    if i == maxiter - 1:
        print("Attention: nombre d'itérations dépassées. La projection de Sammon n'a peut-être pas convergé...")

    # Affichage de la projection
    plt.scatter(
        y[target == 0, 0], y[target == 0, 1], s=20, c="r", marker="o", label=names[0]
    )
    plt.scatter(
        y[target == 1, 0], y[target == 1, 1], s=20, c="b", marker="D", label=names[1]
    )
    plt.scatter(
        y[target == 2, 0], y[target == 2, 1], s=20, c="y", marker="v", label=names[2]
    )
    plt.title("Carte de Sammon des données Iris")
    plt.legend(loc=2)
    plt.show()


if __name__ == "__main__":
    main()