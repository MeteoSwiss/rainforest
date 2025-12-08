from numba import njit


@njit
def add_at_64(arr, idx, toadd):
    N = toadd.shape[0]
    for k in range(N):
        arr[idx[0, k], idx[1, k]] = arr[idx[0, k], idx[1, k]] + toadd[k]
    return arr


@njit
def add_at_32(arr, idx, toadd):
    N = toadd.shape[0]
    for k in range(N):
        arr[idx[0, k], idx[1, k]] = arr[idx[0, k], idx[1, k]] + toadd[k]
    return arr


@njit
def add_at_int(arr, idx, toadd):
    N = toadd.shape[0]
    for k in range(N):
        arr[idx[0, k], idx[1, k]] = arr[idx[0, k], idx[1, k]] + toadd[k]
    return arr
