#cython: language_level=3
# --compile-args="-O2"


cimport cython
import numpy as np

cimport numpy as cnp
from numpy cimport npy_intp, npy_longdouble
from libcpp cimport bool
from libc.math cimport exp



@cython.boundscheck(False)
cdef int calc_dE_2(npy_intp site,
                   cnp.int32_t[::1] spins,          # the field
                   cnp.int32_t[:, :] neighbors):    # associations: neighbors[site, :] are local neighb sites
    """Calculate dE: the energy change for spins[site] -> new_val."""
    
    cdef:
        int old_val = spins[site]
        int this_val
        int e = 0
        npy_intp site1
           
    for j in range(4):    # FIXME: named constant D=2, DD = 2*d
        site1 = neighbors[site, j]
        this_val = spins[site1]
        e += old_val*this_val 
    return 2*e

@cython.cdivision(True)
cdef bint mc_choice(int dE, double T, npy_longdouble uni):
    """принимаем или не принимаем переворот спина?"""
    cdef double r
    r = exp(-dE/T)
    if dE <= 0:
        return True
    elif uni <= r:
        return True
    else:
        return False


@cython.boundscheck(False)
cdef void step(cnp.int32_t[::1] spins, cnp.int32_t[:, :] neigh, double T, int site, npy_longdouble uni):
    """крутим 1 спин"""

        
    cdef int q, L2, dE
    
    L2 = spins.shape[0]
    
    dE = calc_dE_2(site, spins, neigh)
    
    if mc_choice(dE, T, uni):
        spins[site] *= -1
        
    

@cython.boundscheck(False)
def mc_step(cnp.int32_t[::1] spins,
            cnp.int32_t[:, :] neighbors,
            double T):
    """perform L*L flips for 1 MC step"""
    
    cdef npy_intp num_steps = spins.shape[0]
    cdef cnp.ndarray[double,
                ndim=1,
                negative_indices=False,
                mode='c'] unis = np.random.uniform(size=num_steps)
    cdef cnp.ndarray[npy_intp,
                ndim=1,
                negative_indices=False,
                mode='c'] sites = np.random.randint(num_steps, size=num_steps)
    
    for _ in range(num_steps):
        step(spins, neighbors, T, sites[_], unis[_])



    
cdef int kron_c(int i,int j):
    """Kroneker's symbol"""
    if i==j: return 1
    else: return 0    
        

@cython.boundscheck(False)  
# @cython.cdivision(True)
cdef double calc_e_c(cnp.int32_t[::1] spins,
                  cnp.int32_t[:, :] neighbors):
    cdef npy_intp L2 = spins.shape[0]
    cdef int site,j,idx
    cdef int E = 0
    cdef double r
    
    for site in range(L2):
        for j in range(2):
            idx = neighbors[site, j]
            E += kron_c(spins[site],spins[idx])
    r = -E/L2
    return r

def calc_e2(cnp.int32_t[::1] spins,
                  cnp.int32_t[:, :] neighbors):
    cdef double E
    E = calc_e_c(spins, neighbors)
    return E


@cython.boundscheck(False)  
# @cython.cdivision(True)
cdef double calc_m_c(cnp.int32_t[::1] spins):
    
    cdef cnp.ndarray[npy_intp,
                ndim=1,
                negative_indices=False,
                mode='c'] counts = np.zeros(4, dtype=np.intp)
    
    cdef npy_intp L2 = spins.shape[0]
    cdef int site, idx
    cdef double r
    
    for site in range(L2):
        idx = spins[site]
        counts[idx] += 1 
    r = (max(counts)*4/L2 - 1)/3
    return r

def calc_m2(cnp.int32_t[::1] spins):
    cdef double M
    M = calc_m_c(spins)
    return M
