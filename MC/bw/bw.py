#Uses python3

import numpy as np
import pandas as pd 
import time
import sys
import os
import spi3n.mc.bw.cy as cy

#====================== all functions ======================
# neighbours  =============================================
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
        nei += [s[get(i-1),get(j-1)],s[get(i-1),get(j)],s[get(i),get(j+1)],
                s[get(i+1),get(j+1)],s[get(i+1),get(j)],s[get(i),get(j-1)]]
    return np.array(nei, dtype=np.int32).reshape(L*L,6)

# calculations ==================================================================

def create_mask():
    """маска в виде 3 под-решёток"""
    a = np.asarray([i % 3 for i in range(L)])
    return (a + a[:, None])%3

def calc_e(st):
    st = st.reshape(L,L)
    """calculate energy per site
        # expland state matrix"""
    a = np.concatenate((st[L-1].reshape(1,L), st, st[0].reshape(1,L)), axis=0) 
    b = np.concatenate((a[:,-1].reshape(L+2,1),a,a[:,0].reshape(L+2,1)), axis=1)
    return -np.sum(b[1:-1, 1:-1]*b[2:, 2:]*(b[2:, 1:-1]+b[1:-1, 2:]))/(L*L)  

def calc_ms(st):
    """magnetization"""
    st = st.reshape(L,L)
    msr = np.array([np.sum(st[mask==i]) for i in [0,1,2]])/(L*L)
    return np.sqrt(np.sum(msr*msr))

# model ====================================================================

def gen_state():
    """generate random init. state with lenght L*L and q=[-1,1]"""
    return np.array([np.random.choice([-1,1]) for _ in range(L*L)], dtype=np.int32)
        
def model(T,path,N_avg=10,N_mc=10,Relax=10):
    """BW main"""

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


# rest ======================================================================================

def get_t_range(filepath):
    with open(filepath, 'r+') as f:
        t_txt = f.read()
    t_float = list(map(lambda x: float(x.strip()), t_txt.split(',')))
    t_float.sort()
    return t_float


if __name__ == '__main__':

    global L, mask
    L = int(sys.argv[1])               # set size L
    t_path = str(sys.argv[2])          # set path to file with array of T
    t_idx = int(sys.argv[3])           # set index of T_array
    N_img = int(sys.argv[4])           # set number of images you want
    
    mask = create_mask()
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
