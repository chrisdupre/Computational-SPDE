import numpy as np
import scipy as sp
from typing import Tuple


def get_elt_arrays(h:float,p:np.array,
                   q:np.array,f:np.array,
                   ne:int)->Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Takes in a set of p,q,f vectors represnenting difussion, advection
    and forcing terms respectively. Returns a tuple of tensors representing
    the difussion matrices, mass matrices, and forcing vectors associated to 
    each element. 

    Parameters:
    -----------
    h: float
        Length of the element in the natural units of the problem.
    p: np.array
        Array of diffussion coefficients for each element.
    q: np.array
        Array of advection coefficients for each element. 
    f: np.array
        Array of forcing terms for each element. 
    ne: int
        Number of elements. 

    Returns:
    --------
    Kks : np.ndarray
        Tensor where the first index determines the element, and the
        next two correspond to the difussion matrix associated with that 
        element. 
    Mks: np.ndarray
        Tensor where the first index determines the element, and the
        next two correspond to the difussion matrix associated with that 
        element. 
    bks: np.ndarray
        Tensor where the first index determines the element, and the
        next one corresponds to the contribution due to the left and 
        right terms respectively. 

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
                              right_val:float=0
                              )->Tuple[np.array,np.array]:
    """
    A 1D finite element with tent functions
    method solver for general advection-difussion 
    equations with Dirichlet boundrary conditions
    in divergence form:

        -d/dx(p(x)du/dx)+q(x)u(x) = f(x) on x\in (a,b)
        u(a) = left_val
        u(b) = right_val

    Takes the number of elements you would wish to use
    and returns the points determined and the value of 
    the solution at those points. 

    Parameters:
    ----------
    ne: int
        Number of elements. 
    p: np.array
        Array of diffussion coefficients for each element.
    q: np.array
        Array of advection coefficients for each element. 
    f: np.array
        Array of forcing terms for each element. 
    a: float
        Left end point of the interval the function is to be 
        defined on.
    b: float 
        Right end point of the interval the function is to be 
        defined on.
    left_val: float
        Value the function must take on the left end point.
    right_val: float
        Value the function must take on the right end point. 

    Returns:
    --------
    xx: np.array
        Points in the interval the function was determined on
    u: np.array
        Values of the solution at the corresponding points. 
    
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
    u_int = np.linalg.solve(A[1:-1,1:-1].todense(),modified_b[1:-1])
    u = np.zeros(nvtx)
    u[1:-1] = u_int
    u[0] = left_val
    u[-1] = right_val

    return xx, u

def uniform_mesh_info(num_squares:int,
                      a:float = 0,
                      b:float = 1
                      )-> Tuple[np.array,np.array,
                                np.ndarray,int,
                                int,float]:
    """
    Generates a uniform mesh on the 
    square
        [a,b]\times [a,b]
    
    Parameters:
    -----------
    num_squares:int
        Will generate num_squares^2 total
        squares. Input this way so the user 
        doesn't have to check their input is
        square. 
    a:float
        Left end point of interval. Default is 0
    b: float
        Right end point of interval. Default is 1. 

    Returns:
    --------
    xv: np.array
        Array of x coordinates
    yv: np.array
        Array of y coordinates
    elt2vert: np.ndarray
        Array where the i-th row are 
        the labels of the vertices for the
        i-th element
    nvtx: int
        Number of vertices
    ne: int
        Number of elements
    h: float
        Height of uniform mesh
    """
    if a>= b:
        raise ValueError("b must be greater than a.")
    #Construct grid
    h = (b-a)/num_squares
    x = np.arange(a,b+h,h)
    y = x.copy()
    xv, yv = np.meshgrid(x,y)
    xv = xv.flatten()
    yv = yv.flatten()
    #Compute Size of Grid Objects
    nvtx = np.power(num_squares+1,2)
    ne = 2*np.power(num_squares,2)
    #Vertex Labels
    elt2vert = np.zeros((ne,3),dtype=int)
    vv = np.reshape(range(nvtx),
                    (num_squares+1,num_squares+1),
                    order = 'F')
    v1 = vv[:num_squares,:num_squares].flatten(order='F')
    v2 = vv[1:,:num_squares].flatten(order='F')
    v3 = vv[:num_squares,1:].flatten(order='F')
    elt2vert[:np.power(num_squares,2),:] = np.stack((v1,v2,v3),axis=1)
    v4 = vv[1:,1:].flatten(order='F')
    elt2vert[np.power(num_squares,2):,:] = np.stack((v4,v3,v2),axis=1)
    return xv, yv, elt2vert, nvtx, ne, h 


    

    