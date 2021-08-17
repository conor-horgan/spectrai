import torch
from torch import nn
from torch.nn.modules import pooling
from spectrai.networks.layers import BasicConv
from spectrai.networks.layer_utils import get_normalization, get_pooling

# ------------------------------------------------------------
#                           ResNet
# ------------------------------------------------------------
# This code is adapted from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# The corresponding paper is:
# Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun
# Deep Residual Learning for Image Recognition
# IEEE Conference on Computer Vision and Pattern Recognition, 2016
# Available from: https://arxiv.org/abs/1512.03385

def conv3x3(dims, in_planes, out_planes, stride=1, groups=1, dilation=1, 
            normalization = 'BatchNorm', activation = False):
    """3x3 convolution with padding"""
    return BasicConv(dims, in_planes, out_planes, kernel_size = 3,
                    stride = stride, padding = dilation, groups = groups, 
                    bias = False, dilation = dilation, 
                    normalization = normalization, activation = activation)

def conv1x1(dims, in_planes, out_planes, stride=1,
            normalization = 'BatchNorm', activation = False):
    """1x1 convolution"""
    return BasicConv(dims, in_planes, out_planes, kernel_size = 1,
                stride = stride, bias = False, normalization = normalization, 
                activation = activation)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, dims, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, normalization = 'BatchNorm', activation = nn.ReLU()):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.dims = dims

        self.conv1 = conv3x3(self.dims, inplanes, planes, stride, normalization = normalization)
        self.act = activation
        self.conv2 = conv3x3(self.dims, planes, planes, normalization = normalization)

        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, dims, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, normalization = 'BatchNorm', activation = nn.ReLU()):
        super(Bottleneck, self).__init__()
        self.dims = dims
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(self.dims, inplanes, width, normalization = normalization)
        self.conv2 = conv3x3(self.dims, width, width, stride, groups, dilation, normalization = normalization)
        self.conv3 = conv1x1(self.dims, width, planes * self.expansion, normalization = normalization)

        self.act = activation
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.act(out)

        out = self.conv2(out)
        out = self.act(out)

        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out

class ResNet(nn.Module):
    def __init__(self, dims, block, layers, img_channels = 3, res_channels = 64, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, normalization = 'BatchNorm', activation = nn.ReLU()):
        super(ResNet, self).__init__()
    
        self.dims = dims
        self.inplanes = res_channels
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.normalization = normalization
        self.activation = activation

        self.conv1 = BasicConv(self.dims, img_channels, self.inplanes, kernel_size = 7, 
                                stride = 2, padding = 3, bias = False, 
                                normalization = self.normalization)

        self.act = activation
        self.maxpool = get_pooling('MaxPool', size = 3, stride = 2, padding = 1, dims = self.dims)
        self.avgpool = get_pooling('AdaptiveAvgPool', size = 1, dims = self.dims)
        
        self.layer1 = self._make_layer(block, res_channels, layers[0])
        self.layer2 = self._make_layer(block, res_channels*2, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, res_channels*4, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, res_channels*8, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.fc = nn.Linear(res_channels*8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, ((nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                nn.GroupNorm, nn.LayerNorm, 
                                nn.InstanceNorm1d,  nn.InstanceNorm2d,  nn.InstanceNorm3d))):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = conv1x1(self.dims, self.inplanes, planes * block.expansion, stride, normalization = self.normalization)

        layers = []
        layers.append(block(self.dims, self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, normalization = self.normalization, activation = self.activation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.dims, self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                normalization = self.normalization, activation = self.activation))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def _ResNet(dims, block, layers, **kwargs):
    model = ResNet(dims, block, layers, **kwargs)
    return model

def ResNet18(dims, **kwargs):
    return _ResNet(dims, BasicBlock, [2, 2, 2, 2], **kwargs)

def ResNet34(dims, **kwargs):
    return _ResNet(dims, BasicBlock, [3, 4, 6, 3], **kwargs)

def ResNet50(dims, **kwargs):
    return _ResNet(dims, Bottleneck, [3, 4, 6, 3], **kwargs)

def ResNet101(dims, **kwargs):
    return _ResNet(dims, Bottleneck, [3, 4, 23, 3], **kwargs)

def ResNet152(dims, **kwargs):
    return _ResNet(dims, Bottleneck, [3, 8, 36, 3], **kwargs)
    