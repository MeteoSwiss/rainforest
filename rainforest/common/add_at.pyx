import numpy
cimport numpy
cimport cython

ctypedef numpy.float_t DTYPE_t
ctypedef numpy.int_t DTYPE_I
ctypedef numpy.float32_t DTYPE_t32

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def add_at_64(numpy.ndarray[DTYPE_t, ndim=2] arr, numpy.ndarray[DTYPE_I, ndim=2] idx, numpy.ndarray[DTYPE_t, ndim=1] toadd):
    cdef int k  
    cdef int N = toadd.shape[0]  
    for k in range(N):
        arr[idx[0,k],idx[1,k]] = arr[idx[0,k],idx[1,k]] + toadd[k]
    return arr

def add_at_32(numpy.ndarray[DTYPE_t32, ndim=2] arr, numpy.ndarray[DTYPE_I, ndim=2] idx, numpy.ndarray[DTYPE_t32, ndim=1] toadd):
    cdef int k  
    cdef int N = toadd.shape[0]  
    for k in range(N):
        arr[idx[0,k],idx[1,k]] = arr[idx[0,k],idx[1,k]] + toadd[k]
    return arr

def add_at_int(numpy.ndarray[DTYPE_I, ndim=2] arr, numpy.ndarray[DTYPE_I, ndim=2] idx, numpy.ndarray[DTYPE_I, ndim=1] toadd):
    cdef int k  
    cdef int N = toadd.shape[0]  
    for k in range(N):
        arr[idx[0,k],idx[1,k]] = arr[idx[0,k],idx[1,k]] + toadd[k]
    return arr
