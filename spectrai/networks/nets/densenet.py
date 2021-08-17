import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor
from typing import Any, List, Tuple

from spectrai.networks.layers import BasicConv
from spectrai.networks.layer_utils import get_normalization, get_pooling

# ------------------------------------------------------------
#                           DenseNet
# ------------------------------------------------------------
# This code is adapted from: https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
# The corresponding paper is:
# Gao Huang, Zhuang Liu, Laurens van der Maaten
# Deeply Connected Convolutional Networks
# IEEE Conference on Computer Vision and Pattern Recognition, 2017
# Available from: https://arxiv.org/abs/1608.06993

class _DenseLayer(nn.Module):
    def __init__(self, dims, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient = False,
                normalization = 'BatchNorm', activation = nn.ReLU(inplace=True)):
        super(_DenseLayer, self).__init__()
        self.norm1 = get_normalization(num_input_features, normalization, dims)
        self.act1 = activation
        self.conv1 = BasicConv(dims, num_input_features, bn_size *
                                growth_rate, kernel_size=1, stride=1,
                                bias=False, normalization = None,
                                activation = None)
        self.norm2 = get_normalization(bn_size * growth_rate, normalization, dims)
        self.act2 = activation
        self.conv2 = BasicConv(dims, bn_size * growth_rate, growth_rate,
                                kernel_size=3, stride=1, padding=1,
                                bias=False, normalization = None,
                                activation = None)
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.act1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    def any_requires_grad(self, input: List[Tensor]) -> bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input: List[Tensor]) -> Tensor:
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: List[Tensor]) -> Tensor:
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: Tensor) -> Tensor:
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.act2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features

class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, dims, num_layers, num_input_features, bn_size, growth_rate,
        drop_rate, memory_efficient= False, normalization = 'BatchNorm', 
        activation = nn.ReLU(inplace=True)):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                dims,
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                normalization=normalization,
                activation=activation
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

class _Transition(nn.Sequential):
    def __init__(self, dims, num_input_features, num_output_features,
                normalization = 'BatchNorm', activation = nn.ReLU(inplace=True)):
        super(_Transition, self).__init__()
        self.norm = get_normalization(num_input_features, normalization, dims)
        self.act = activation
        self.conv = BasicConv(dims, num_input_features, num_output_features,
                            kernel_size=1, stride=1, bias=False, 
                            normalization=None, activation=None)
        self.pool = get_pooling('AvgPool', size = 2, stride = 2, dims = dims)

class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """

    def __init__(self, dims, channels = 3, growth_rate = 32, block_config = (6, 12, 24, 16),
        num_init_features = 64, bn_size = 4, drop_rate = 0, num_classes = 1000,
        memory_efficient = False, normalization = 'BatchNorm', activation = nn.ReLU(inplace=True)):
        super(DenseNet, self).__init__()
        
        self.dims = dims

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', BasicConv(dims, channels, num_init_features, kernel_size=7,
                        stride=2, padding=3, bias=False,
                        normalization=normalization, activation=None)),
            ('relu0', activation),
            ('pool0', get_pooling('MaxPool', size=3, stride=2, padding=1, dims=dims)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                dims=dims,
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                normalization=normalization,
                activation=activation
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(dims = dims, num_input_features=num_features,
                                    num_output_features=num_features // 2,
                                    normalization=normalization,
                                    activation=activation)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', get_normalization(num_features, normalization, dims))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        if self.dims == 1:
            out = F.adaptive_avg_pool1d(out, 1)
        elif self.dims == 2:
            out = F.adaptive_avg_pool2d(out, 1)
        else: #self.dims == 3:
            out = F.adaptive_avg_pool3d(out, 1)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

def _densenet(arch, dims, channels, growth_rate, block_config, num_init_features, **kwargs):
    model = DenseNet(dims, channels, growth_rate, block_config, num_init_features, **kwargs)
    return model

def densenet121(dims, channels, **kwargs) -> DenseNet:
    return _densenet('densenet121', dims, channels, 32, (6, 12, 24, 16), 64, **kwargs)

def densenet161(dims, channels, **kwargs) -> DenseNet:
    return _densenet('densenet161', dims, channels, 48, (6, 12, 36, 24), 96, **kwargs)

def densenet169(dims, channels, **kwargs) -> DenseNet:
    return _densenet('densenet169', dims, channels, 32, (6, 12, 32, 32), 64, **kwargs)

def densenet201(dims, channels, **kwargs) -> DenseNet:
    return _densenet('densenet201', dims, channels, 32, (6, 12, 48, 32), 64, **kwargs)
