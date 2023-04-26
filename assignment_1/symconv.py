import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from typing import Callable



class SymmetryPool(nn.Module):
    """
    Performs pooling over different symmetries
    Takes as input pooling function
    Pooling function should only return 1 value (use torch.amax instead of torch.max)
    
    Parameters
    -----------------------
        pool (Callable): function used for pooling (default: torch.amax)
    -----------------------
    in shape: B, S, F, W, H
    out shape: B, F, W, H
    -----------------------
    B: batch size
    S: amount of symmetries
    F: amount of filters
    W: width
    H: height
    """
    def __init__(self, pool : Callable = torch.amax):
        
        super().__init__()
        
        self.pool = pool
        
    def forward(self, x):
        
        return self.pool(x, dim=1)
    
    

class Slice(nn.Module):
    """
    Creates copies of the feature map for rotated and/or reflected kernels
    
    Parameters
    -----------------------
        rotation (int): amount of rotations that will be performed (can be either 1, 2 or 4)
        reflection (bool): bool to indicate whether reflections will be performed
    -----------------------
    in shape: B, F, W, H
    out shape: B, S, F, W, H
    -----------------------
    B: batch size
    S: amount of symmetries
    F: amount of filters
    W: width
    H: height
    """
    def __init__(self,
                 rotation : int,
                 reflection: bool):
        
        super().__init__()
        
        assert rotation in [1,2,4]
        
        self.mul = rotation*(1+1*reflection)
        
    def forward(self, x):
        
        assert len(x.shape) == 4
        
        # add dimension for each symmetry
        x = x[:,None].repeat(1,self.mul,1,1,1)
        
        return x

    



class SymmetryConv2d(nn.Module):
    """
    Conv2d layer where kernel is rotated/reflected for each symmetry
    
    For example, if rotation is 4, during the forward call a feature map will be produced for each of the 4 possible rotations of the kernel
    
    Parameters
    -----------------------
        in_filters (int): amount of input filters
        out_filters (int): amount of output filters
        kernel_size (int): size of the kernel
        stride (int): size of the stride
        bias (bool): whether bias should be included in the layer
        rotation (int): amount of rotations that will be performed (can be either 1, 2 or 4)
        reflection (bool): bool to indicate whether reflections will be performed
    -----------------------
    in shape: B, S, F$_{in}$, W, H
    out shape: B, S, F$_{out}$, W, H
    -----------------------
    B: batch size
    S: amount of symmetries
    F$_{in}$: amount of input filters
    F$_{out}$: amount of output filters
    W: width
    H: height
    """
    
    def __init__(self,
                 in_filters : int,
                 out_filters : int,
                 kernel_size : int,
                 stride : int = 1,
                 bias : bool = True,
                 rotation : int = 2,
                 reflection : bool = True,
        ):
        
        super().__init__()
        
        assert rotation in [1,2,4]
        
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.rotation = rotation
        self.reflection = reflection
        
        self.weight = nn.Parameter(torch.randn((self.out_filters, self.in_filters, kernel_size, kernel_size)))
        
        self.bias = nn.Parameter(torch.randn((self.out_filters))) if bias else None
        
    
    
    def get_out_shape(self, x):
        
        H = int(np.floor((x.shape[-2] - (self.kernel_size - 1) - 1)/self.stride + 1))
        W = int(np.floor((x.shape[-1] - (self.kernel_size - 1) - 1)/self.stride + 1))
        
        return H, W
        
    def forward(self, x, H=None, W=None):
        
        assert len(x.shape) == 5
        assert x.shape[1] == (1+1*self.reflection)*self.rotation
        
        if H is None or W is None:
            H, W = self.get_out_shape(x)
        
        
        out_tensor = torch.zeros((x.shape[0], x.shape[1], self.out_filters, H, W), device=x.device)
        
        
        out_tensor[:,0] = F.conv2d(x[:,0], self.weight, self.bias, self.stride)
        
        if self.reflection:
            out_tensor[:,1] = F.conv2d(x[:,0], torch.flip(self.weight, dims=(2,)), self.bias, self.stride)
        
        for k in range(1, self.rotation):
            k_rot = k if self.rotation == 4 else 2
            
            
            w_rot = torch.rot90(self.weight, dims=(2,3), k=k_rot)
            
            if self.reflection:
                out_tensor[:,2*k] = F.conv2d(x[:,2*k], w_rot, self.bias, self.stride)
                out_tensor[:,2*k+1] = F.conv2d(x[:,2*k+1], torch.flip(w_rot, dims=(2,)), self.bias, self.stride)
            
            else:
                out_tensor[:,k] = F.conv2d(x[:,k], w_rot, self.bias, self.stride)
            
            
        
        return out_tensor
            