import numpy as np
import scipy as sp

"""
Shared library of functions for functions used across exercises. 

@author: Christopher DuPre

"""

def DST_1(x:np.array)->np.array:
    """
    FFT computation of DST-1
    using  scipy

    Parameters:
    ----------
    x: np.array
        Array to compute DCT-1 on 

    Returns:
    --------
    y: np.array
        DCT-1 computed Array
    
    """
    N= len(x)
    #Generate vector of length 2N+2 for DFT
    fft_vect = np.zeros(2*N+2,dtype=complex)
    fft_vect[1:N+1] = x
    # [::-1] is just a fancy string indexing way of writing reverse
    fft_vect[N+2:2*N+2] = -x[::-1]
    y = 1.j/2*(sp.fft.fft(fft_vect))[1:N+1]
    return y

def DCT_1(x:np.array)->np.array:
    """
    Computation of DCT-1 using 
    FFT given by scipy

    Parameters:
    ----------
    x: np.array
        Array to compute DCT-1 on 

    Returns:
    --------
    y: np.array
        DCT-1 computed Array
    
    """
    N= len(x)
    #Generate vector of length 2N-2 for FFT
    fft_vect = np.zeros(2*N-2)
    fft_vect[0] = x[0]/2
    fft_vect[N-1] = x[N-1]/2
    fft_vect[1:N-1] = x[1:N-1]/2
    # [::-1] is just a fancy string indexing way of writing reverse
    fft_vect[N:2*N-2] = x[1:N-1][::-1]/2
    y = sp.fft.fft(fft_vect)[:N]
    return y


