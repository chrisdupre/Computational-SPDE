import numpy as np
import scipy as sp
from typing import Tuple


def get_elt_arrays(h:float,p:np.array,
                   q:np.array,f:np.array,
                   ne:int)->Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """


    Parameters:
    -----------

    
    Returns:
    --------


    """
    #Initialize elements 
    Kks = np.zeros((ne,2,2))
    Mks =  np.zeros((ne,2,2))
    bks = np.zeros((ne,2))
    #Form Difussion matrices
    Kks[:,0,0] = p/h
    Kks[:,0,1] = -p/h
    Kks[:,1,0] = -p/h
    Kks[:,1,1] = p/h
    #Form mass matrices
    Mks[:,0,0] = q*h/3
    Mks[:,0,1] = q*h/6
    Mks[:,1,0] = q*h/6
    Mks[:,1,1] = q*h/3
    #Form bks
    bks[:,0] = f*h/2
    bks[:,1] = f*h/2
    return Kks, Mks, bks

def oned_linear_FEM_Dirichlet(ne:int,
                              p:np.array,
                              q:np.array,
                              f:np.array,
                              a:float=0,
                              b:float=1, 
                              left_val:float=0,
                              right_val:float=0):
    """

    Parameters:
    ----------

    Returns:
    --------
    
    """
    if b<=a:
        raise ValueError("b must be greater than a.")
    if not(p.ndim == 1 and q.ndim == 1 and f.ndim == 1):
        raise ValueError("p,q,h must all be vectors.")
    if not(len(p)==ne and len(q)==ne and len(f)==ne):
        raise ValueError("All arrays must be of length ne.")
    
    h = (b-a)/ne
    xx = np.arange(a,b+h,h)
    nvtx = len(xx)
    K = sp.sparse.lil_array((nvtx,nvtx))
    M = sp.sparse.lil_array((nvtx,nvtx))
    b = np.zeros(nvtx)
    Kks, Mks, bks = get_elt_arrays(h = h,
                                   p = p,
                                   q = q,
                                   f = f,
                                   ne = ne)

    for count in range(ne):
        K[count:count+2,count:count+2] += Kks[count,:,:]
        M[count:count+2,count:count+2] += Mks[count,:,:]
    b[:ne] = bks[:,0]
    b[-ne:] = bks[:,1]

    #Impose homogenous boundrary conditions
    A = K+M
    modified_b = b-left_val*A.getcol(0).todense().flatten()-right_val*A.getcol(-1).todense().flatten()
    print(modified_b.shape)
    u_int = np.linalg.solve(A[1:-1,1:-1].todense(),modified_b[1:-1])
    u = np.zeros(nvtx)
    u[1:-1] = u_int
    u[0] = left_val
    u[-1] = right_val

    return xx, u

    

    