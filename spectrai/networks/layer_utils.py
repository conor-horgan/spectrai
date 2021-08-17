import torch
from torch import nn

def get_normalization(channels, normalization, dims):
    """Returns the appropriate normalization layer given channels, dims.

    Arguments:
        channels: number of layer channels for normalization
        normalization: desired normalization layer (str)
        dims: data dimensions (1D, 2D, or 3D)
    
    Returns:
        layers: list of PyTorch neural network layers
    """
    if normalization == 'BatchNorm':
        if dims == 1:
            return nn.BatchNorm1d(channels)
        elif dims == 2:
            return nn.BatchNorm2d(channels)
        elif dims == 3:
            return nn.BatchNorm3d(channels)
    elif normalization == 'InstanceNorm':
        if dims == 1:
            return nn.InstanceNorm1d(channels)
        elif dims == 2:
            return nn.InstanceNorm2d(channels)
        elif dims == 3:
            return nn.InstanceNorm3d(channels)
    elif normalization == 'LayerNorm':
        return nn.LayerNorm(channels)
    elif normalization == 'GroupNorm':
        return nn.GroupNorm(channels/4, channels)
    else:
        return None
    
def get_pooling(pooling, size, stride = None, padding = 0, dilation = 1, dims = 2):
    """Returns the appropriate pooling layer given pooling, dims.

    Arguments:
        pooling: desired pooling layer (str)
        size: kernel size or output size
        dims: data dimensions (1D, 2D, or 3D)
    
    Returns:
        layers: list of PyTorch neural network layers
    """
    if pooling == 'MaxPool':
        if dims == 1:
            return nn.MaxPool1d(size, stride = stride, padding = padding, dilation = dilation)
        elif dims == 2:
            return nn.MaxPool2d(size, stride = stride, padding = padding, dilation = dilation)
        elif dims == 3:
            return nn.MaxPool3d(size, stride = stride, padding = padding, dilation = dilation)
    elif pooling == 'AvgPool':
        if dims == 1:
            return nn.AvgPool1d(size, stride = stride, padding = padding)
        elif dims == 2:
            return nn.AvgPool2d(size, stride = stride, padding = padding)
        elif dims == 3:
            return nn.AvgPool3d(size, stride = stride, padding = padding)
    elif pooling == 'AdaptiveAvgPool':
        if dims == 1:
            return nn.AdaptiveAvgPool1d(size)
        if dims == 2:
            return nn.AdaptiveAvgPool2d(size)
        if dims == 3:
            return nn.AdaptiveAvgPool3d(size)
    else:
        return None