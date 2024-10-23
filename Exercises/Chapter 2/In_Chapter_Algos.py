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

def get_jac_info(xv:np.array,
                 yv:np.array,
                 ne:int,
                 elt2vert:np.ndarray
                 )->Tuple[np.ndarray,
                          np.ndarray,
                          np.array]:
    """
    Get Jacobian information corresponding
    to a simple triangulation of [0,1]^2

    Parameters:
    ------------
    xv: np.array 
        Grid of x points
    yv: np.array
        Grid of y points
    ne: int
        Number of finite elements
    elt2vert: np.ndarray
        Maps elements to the three 
        indices of its corresponding
        vertices.

    Returns:
    --------
    Jks: np.ndarray (ne,2,2)
        Tensor of Jaobian matrices for 
        transformation of finite elements 
        into standard form. 
    invJks: np.ndarray (ne,2,2)
        Tensor of inverse Jacobain matries
        for transformation of finite elements
        into standard form.
    detJks: np.array (ne)
        Array of determinants 

    """
    #Grab elements corresponding to each finite element
    [x1,x2,x3] = [xv[elt2vert[:,j]] for j in range(3)]
    [y1,y2,y3] = [yv[elt2vert[:,j]] for j in range(3)]

    # Compute Jaobian Matrices
    Jks = np.zeros((ne,2,2))
    Jks[:,0,0] = x2-x1
    Jks[:,0,1] = y2-y1
    Jks[:,1,0] = x3-x1
    Jks[:,1,1] = y3-y1

    #Compute Determinants 
    vetorized_det = np.vectorize(np.linalg.det,
                                 signature="(m,n,k)->(m)")
    detJks = vetorized_det(Jks)

    #Compute Inverse of Jaobians 
    invJks = np.zeros((ne,2,2))
    invJks[:,0,0] = np.multiply((1/detJks),y3-y1)
    invJks[:,0,1] = np.multiply((1/detJks),y1-y2)
    invJks[:,1,0] = np.multiply((1/detJks),x1-x3)
    invJks[:,1,1] = np.multiply((1/detJks),x2-x1)

    return Jks,invJks,detJks


def get_elt_arrays2D(xv:np.array,
                     yv:np.array,
                     invJks:np.ndarray,
                     detJks:np.array,
                     ne:int,
                     elt2vert:np.ndarray,
                     a:np.array,
                     f:np.array
                     )-> Tuple[np.ndarray,
                               np.ndarray]:
    """
    Parameters:
    ---------------
    xv: np.array
        Array of possible x values

    yv: np.array
        Array of possible y values

    invJks: np.ndarray
        Tensor of inverse Jacobian matrices

    detJks: np.array
        Array of determinant elements

    ne: int
        Number of finite elements

    el2vert: np.ndarray
        Array mapping a finite element
        to the indices of its vertices

    a: np.array
        Vector of a at every finite element

    f: np.array
        Vector of f at every finite element

    Returns:
    ---------
    Aks: np.ndarray
        Tensor containing A for each finite element
        (considering pairs of vertices)
    
    bks: np.ndarray
        Array containing the weight vector for each
        vertex in the finite element

    NOTE: xv,yv,elt2vert are not used. The point is to 
    keep a consistent signature for when future schema may
    depend on these elements. This may not be the best
    approach, but I am sticking with the book's schema
    until I am sure how I would prefer it.
    """
    #Preallocate memory
    bks = np.zeros((ne,3))
    Aks = np.zeros((ne,3,3))
    #Generate gradient matrix
    dpsi = np.array([[-1,1,0],
                     [-1,0,1]])
    for i in range(3):
        for j in range(3):
            #Grab Relevant Entrie
            grad = dpsi[:,[i,j]]
            #Compute Gradient in usual coordinates
            v = np.einsum("ijk,kl->ijl",
                          invJks,grad)
            #Compute Dot Product
            integrand = np.sum(np.prod(v,axis=2),
                               axis=1)
            #Multiply to form weights
            Aks[:,i,j] = np.multiply(a,detJks,integrand)/2
        #Find weight vector
        bks[:,i] = np.multiply(f,detJks)/6

    return Aks,bks

def twodlinearFEMDirichlet(ns:int,
                           xv:np.array,
                           yv:np.array,
                           elt2vert:np.ndarray,
                           nvtx:int,
                           ne:int,
                           h:float,
                           a:np.array,
                           f:np.array,
                           )->Tuple[np.array,
                                    np.ndarray,
                                    np.array]:
    """
    Parameters:
    -----------
    ns: int
        Number of squares to provide per side, 
        produces ns^2 actual squares

    xv: np.array
        Array of x coordinates per vertex index.

    yv: np.array
        Array of x coordinates per vertex index.

    elt2vert: np.ndarray
        Array relating the index of a finite element
        to the indices of its vertices

    nvtx: int
        Number of vertices

    ne: int
        Number of finite elements

    h: float
        Step size of mesh

    a: np.array
        Array of a per finite element

    f: np.array
        Array of forcing term per finite
        element.

    Returns:
    --------
    u_int: np.array
        Interior solutions
    
    A_int: np.ndarray
        Finite element matrix
    
    rhs: np.array
        Right hand side of solver
    """
    Jks, invJks, detJks = get_jac_info(xv = xv,
                                       yv = yv,
                                       ne = ne,
                                       elt2vert = elt2vert)
    Aks, bks = get_elt_arrays(xv = xv,
                              yv = yv,
                              invJks = invJks,
                              detJks = detJks,
                              ne = ne,
                              elt2vert = elt2vert,
                              a = a,
                              f = f)
    
    A = sp.sparse.lil_array((nvtx,nvtx))
    b = np.zeros(nvtx)
    
    for element in range(ne):
        rel_indices = elt2vert[element,:]
        A[rel_indices,rel_indices] += Aks[element,:,:]
        b[rel_indices] += bks[element,:]

    b_nodes = np.array([index for index in range(nvtx) 
                        if xv[index]==0 or xv[index]==1
                        or yv[index]==0 or yv[index]==1])
    b_interior = np.array([index for index in range(nvtx) 
                           if index not in b_nodes])
    
    A_int = A[b_interior,b_interior]
    rhs = b[b_interior]

    u_int = np.linalg.solve(A = A_int.todense(),
                            b=rhs)
    
    return u_int, A_int, rhs

    