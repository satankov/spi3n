import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.transforms import Compose
import numpy as np
import pandas as pd


class SimpleSampler(torch.utils.data.Dataset):
    """
    data = {
        $T$ : pd.DataFrame
    }
    
    """
    def __init__(self, img_arr, data, transform=None):
        self.t_arr = list(data.keys())
        self.img_arr = img_arr
        self.data = data
        self.transform = transform
        self.meta = self._tabulate_()
        self.L = self._getsize_()
        
    def _getsize_(self):
        for temperature in self.data:
            df_size = self.data[temperature].shape[1]  # 1st col is T  +  L**2 
            L = int(np.sqrt(df_size-1))
            break
        return L
        
    def _tabulate_(self):
        meta = []
        for t in range(len(self.t_arr)):
            for obj in range(len(self.img_arr)):
                meta += [[t, obj]]
        return meta
        
    def _load_(self, T, obj_idx):
        df = self.data.get(T).iloc[obj_idx:obj_idx+1,1:].copy()
        return df

    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, idx):
        t_idx, obj_idx = self.meta[idx]
        T = self.t_arr[t_idx]
        data = self._load_(T, obj_idx)\
            .values.reshape(self.L,self.L)
        return T, self.transform(data)



class TemperatureNegSamplerFile(torch.utils.data.Dataset):
    def __init__(self, t_arr, l_arr, img_arr, path='', window=5, delta=0.0005, transform=None):
        self.t_arr = t_arr
        self.l_arr = l_arr
        self.img_arr = img_arr
        self.path = path
        self.meta = self._tabulate_()
        self.t_neg = self._make_candidates_(t_arr, window, delta)
        self.transform = transform
        
    def _tabulate_(self):
        meta = []
        for t in range(len(self.t_arr)):
            for l in range(len(self.l_arr)):
                for obj in range(len(self.img_arr)):
                    meta += [[t,l,obj]]
        return meta
    
    def _make_candidates_(self, t_arr, window, delta):
        res = {}
        for idx, t in enumerate(t_arr):
            idx_min = max([ idx-window, 0 ])
            idx_max = min([ idx+window, len(t_arr) ])
            t_arr_1 = np.array(t_arr[idx_min:idx_max])
            t_arr_2 = t_arr_1[(t_arr_1 <= t-delta) | (t_arr_1 >= t+delta)]
            res[t] = t_arr_2
        return res
    
    def _get_path_(self, T, L):
        t_str = str(T).replace(".","_")
        path = f"{self.path}samples_{L}/{t_str}.csv"
        return path
        
    def _load_(self, path, obj_idx):
        df = pd.read_csv(
            path, 
            skiprows=obj_idx,
            nrows=1
        )
        return df
        
    def get_anchor(self, idx):
        t_idx, l_idx, obj_idx = self.meta[idx]
        L = self.l_arr[l_idx]
        T = self.t_arr[t_idx]
        IMG = self.img_arr[obj_idx]
        data = self._load_(
            self._get_path_(T, L),
            IMG
        ).iloc[:,1:].values.reshape(L,L)
        return T, self.transform(data)
    
    def get_positive(self, T):
        L_rand = np.random.choice(self.l_arr)
        obj_idx_rand = np.random.choice(self.img_arr)
        data = self._load_(
            self._get_path_(T, L_rand),
            obj_idx_rand
        ).iloc[:,1:].values.reshape(L_rand,L_rand)
        return self.transform(data)
    
    def get_negative(self, T):
        T_rand = np.random.choice(self.t_neg[T])
        L_rand = np.random.choice(self.l_arr)
        obj_idx_rand = np.random.choice(self.img_arr)
        data = self._load_(
            self._get_path_(T_rand, L_rand),
            obj_idx_rand
        ).iloc[:,1:].values.reshape(L_rand,L_rand)
        return self.transform(data)
        
    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, idx):
        T, anchor = self.get_anchor(idx)
        positive = self.get_positive(T)
        negative = self.get_negative(T)
        return T, anchor, positive, negative
    
    

class TemperatureNegSamplerObject(torch.utils.data.Dataset):
    """
    data = {
        $L$: {
            $T$ : pd.DataFrame
        }
    }
    
    
    """
    def __init__(self, t_arr, l_arr, img_arr, data, window=5, delta=0.0005, transform=None):
        self.t_arr = t_arr
        self.l_arr = l_arr
        self.img_arr = img_arr
        self.data = data
        self.meta = self._tabulate_()
        self.t_neg = self._make_candidates_(t_arr, window, delta)
        self.transform = transform
        
    def _tabulate_(self):
        meta = []
        for t in range(len(self.t_arr)):
            for l in range(len(self.l_arr)):
                for obj in range(len(self.img_arr)):
                    meta += [[t,l,obj]]
        return meta
    
    def _make_candidates_(self, t_arr, window, delta):
        res = {}
        for idx, t in enumerate(t_arr):
            idx_min = max([ idx-window, 0 ])
            idx_max = min([ idx+window, len(t_arr) ])
            t_arr_1 = np.array(t_arr[idx_min:idx_max])
            t_arr_2 = t_arr_1[(t_arr_1 <= t-delta) | (t_arr_1 >= t+delta)]
            res[t] = t_arr_2
        return res
        
    def _load_(self, L, T, obj_idx):
        df = self.data.get(L).get(T).iloc[obj_idx:obj_idx+1,1:].copy()
        return df
        
    def get_anchor(self, idx):
        t_idx, l_idx, obj_idx = self.meta[idx]
        L = self.l_arr[l_idx]
        T = self.t_arr[t_idx]
        img_idx = self.img_arr[obj_idx]
        data = self._load_(
            L, T, img_idx
        ).values.reshape(L,L)
        return L, T, self.transform(data)
    
    def get_positive(self, T):
        L_rand = np.random.choice(self.l_arr)
        obj_idx_rand = np.random.choice(self.img_arr)
        data = self._load_(
            L_rand, T, obj_idx_rand
        ).values.reshape(L_rand,L_rand)
        return self.transform(data)
    
    def get_negative(self, T):
        T_rand = np.random.choice(self.t_neg[T])
        L_rand = np.random.choice(self.l_arr)
        obj_idx_rand = np.random.choice(self.img_arr)
        data = self._load_(
            L_rand, T_rand, obj_idx_rand
        ).values.reshape(L_rand,L_rand)
        return self.transform(data)
        
    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, idx):
        L, T, anchor = self.get_anchor(idx)
        positive = self.get_positive(T)
        negative = self.get_negative(T)
        return L, T, anchor, positive, negative
