import torch
import torchvision
import torchvision.transforms.functional as F
import numpy as np


class array2tensor(object):
    def __init__(self):
        pass
    def __call__(self, np_array):
        torch_tensor = torch.from_numpy(np_array)\
            .unsqueeze(0)\
            .long()
        return torch_tensor
    
    
class Encode(object):
    def __init__(self):
        pass
    def __call__(self, sample):
        sample[sample == -1] = 0
        return sample
    
    
class Rotate(object):
    def __init__(self, axis=[1, 2]):
        self.axis = axis
    def __call__(self, sample):
        n_rot = np.random.randint(4)
        sample_rot = torch.rot90(sample, n_rot, self.axis)
        return sample_rot


class Resize(object):
    def __init__(self, output_size):
        self.out_sz = output_size
    def __call__(self, sample):
        out = F.resize(
            sample, 
            size=(self.out_sz,self.out_sz),
            interpolation=torchvision.transforms.InterpolationMode.NEAREST,
        )
        return out
