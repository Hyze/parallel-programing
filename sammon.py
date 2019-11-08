import profile

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn import datasets
from pyopencl.elementwise import ElementwiseKernel
from pyopencl.reduction import ReductionKernel
import pyopencl.array as cl_array
import pyopencl as cl


deviceID = 0
platformID = 1


dev = cl.get_platforms()[platformID].get_devices()[deviceID]

ctx = cl.Context([dev])
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags


kernel_g = cl.Program(ctx,"""__kernel void gradient(__global float  *D, __global float *d, __global float2* g, __global float2* y , int n)
    {
        unsigned int p = get_local_id(0); //get_global_id(0)?
        float2 sum=0;
    
        
        for(unsigned int j=0;j<n;j++)
        {
            if(p!=j)
            {   

                sum+=(D[p*n+j]-d[p*n+j])/(D[p*n+j]*d[p*n+j]) * (y[p]-y[j]);
               //printf(" %f P %d  \\n", (float2) -sum, (int) p);
            }
        }
        g[p]=-sum;
       
           
    }""").build()

kernel_h=cl.Program(ctx,"""
    __kernel void hessian(__global float * D,__global float * d,__global float2 * h,__global float2 * y,const int n)
    {
        uint p=get_local_id(0);
        float2 sum=0;
        for(int j=0;j<n;j++){
            if(p!=j){
                sum+=(1/(D[p*n+j]*d[p*n+j])*((D[p*n+j]-d[p*n+j])-((y[p]-y[j])*(y[p]-y[j]))/d[p*n+j]*(1+(D[p*n+j]-d[p*n+j])/d[p*n+j])));
            }
        }
        h[p]=-sum;
    }

    """
    ).build()






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


def cl_gradient(D, d, y):
    N, M = y.shape
    g = np.zeros([N, M], dtype=np.float64).flatten()

    cl_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=g)
    cl_D = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=D)
    cl_d = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=d)
    cl_y = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=y)

    kernel_g.gradient(queue, (N, 1), None, cl_D, cl_d, cl_g, cl_y, np.int32(N))
    cl.enqueue_copy(queue, g, cl_g)
    return g.flatten().astype(np.float64)


    #Cette solution ne fonctionne pas le compilateur me dit que je redefinis n je comprend pas

    # matrice = eltWise(ctx, '__global float *D,__global float *d,int *n,__global float2 *g',
    #                   '''
    #                 unsigned int p = get_local_id(0);
    #
    #                  float2 sum=0;
    #                  for(unsigned int j=0;j<n;j++){
    #                  unsigned int temp = n ;
    #                      if(p!=j)
    #                     {
    #                         sum += (D[p*temp+j]-d[p*temp+j])/(D[p*temp+j] * d[p*temp+j]);
    #                     }
    #                  }
    #                      g[p]=-sum;
    #
    #               ''', name ='matrice',preamble="", options=[])
    #
    # matrice(cl_D, cl_d,np.int32((N,1)),cl_g)



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
def cl_hessian(D, d, y):
    N, M = y.shape
    h = np.zeros([N,M],dtype=np.float32).flatten()

    cl_h = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h)
    cl_D = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=D)
    cl_d = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=d)
    cl_y = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=y)

    kernel_h.hessian(queue, (N, 1), None, cl_D, cl_d, cl_h, cl_y, np.int32(N))
    cl.enqueue_copy(queue, h, cl_h)
    return h.flatten().astype(np.float32)

def y_update_constant(y, s, D, d, alpha=0.3):
    y = y + alpha * s
    d = cdist(y, y).astype(np.float32)
    E_new = cl_error(D, d)
    return y, E_new


# @profile
def cl_error(D, d):
    cl_D = cl.array.to_device(queue, D)
    cl_d = cl.array.to_device(queue, d)
    krnl = ReductionKernel(ctx, np.float32, neutral="0",
                           reduce_expr="a+b", map_expr="x[i] != 0 ? ((x[i] - y[i])*(x[i] - y[i]))/x[i]: 0.0",
                           arguments="__global float *x, __global float *y")
    res_error = krnl(cl_D, cl_d).get()
    return res_error


def y_update_halving(y, s, D, d, maxhalves=20):
    # E = error(D, d)
    E = cl_error(D, d)
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


# @profile
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

        #g = gradient(D, d, y)
        g = cl_gradient(D, d, y)

        # et de la Hessienne
        #H = hessian(D, d, y)
        H = cl_hessian(D,d,y)
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
