#Uses python3

import numpy as np
import pandas as pd
import time
import sys, os
import spi3n.mc.ising.cy as cy

#====================== all functions ======================

def coord(site):
    """get coordinate i of vector"""
    x = site // L
    y = site - x*L
    return (x,y)

def get(i):
    """fixin' boundary"""
    if i<0: return i
    else: return i % L

def get_neigh():
    """get neighbour's arr"""
    s = np.arange(L**2).reshape(L,L)
    nei = []
    for site in range(L*L):
        i,j = coord(site)
        nei += [s[get(i-1),get(j)],s[get(i),get(j+1)],s[get(i+1),get(j)],s[get(i),get(j-1)]]
    return np.array(nei, dtype=np.int32).reshape(L*L,4)

#################################################################

def gen_state():
    """generate random start state with lenght L*L and q components"""
    state = np.array([np.random.choice([-1,1]) for _ in range(L*L)], dtype=np.int32)
    return state

################################################################################

def model(T,path,N_avg=10,N_mc=10,Relax=10):
    """Ising main"""

    state = gen_state()
    nei = get_neigh()
    
    size = state.shape[0]
    cols = ['T']+[_ for _ in range(size)]
    df_init = pd.DataFrame(data=[], columns=cols, )
    df_init.to_csv(path, index=None, header=True)

    #relax $Relax times be4 AVG
    for __ in range(Relax):
        cy.mc_step(state, nei, T)
    #AVG every $N_mc steps
    for _ in range(N_avg):
        for __ in range(N_mc):
            cy.mc_step(state, nei, T)
        df = pd.DataFrame(data=[state],)
        df.insert(0, 'T', T)
        df.to_csv(path, index=None, header=False, mode='a')


def get_t_range(filepath):
    with open(filepath, 'r+') as f:
        t_txt = f.read()
    t_float = list(map(lambda x: float(x.strip()), t_txt.split(',')))
    t_float.sort()
    return t_float



#################################################################

if __name__ == '__main__':

    global L
    L = int(sys.argv[1])               # set size L
    t_path = str(sys.argv[2])          # set path to file with array of T
    t_idx = int(sys.argv[3])           # set index of T_array
    N_img = int(sys.argv[4])           # set number of images you want
    
    seed = 0+t_idx
    np.random.seed(seed)
    tc = 1/(np.log(2**0.5+1)/2)        # 2.269185314213022
    T = get_t_range(t_path)[t_idx]
    
    z = 2.15
    Relax = int(12*L**z)
    N_mc = int(2*L**z)

    title = str(T).replace(".",'_') + '.csv'
    out_path = f"data/samples_{L}/"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    model(T, out_path+title, N_img, N_mc, Relax)      
