from spectrai.networks.layers import BasicConv
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
#                           SegNet
# ------------------------------------------------------------
# This code is adapted from: https://github.com/trypag/pytorch-unet-segnet
# The corresponding paper is:
# Vijay Badrinarayanan, Alex Kendall, and Roberto Cipolla 
# Segnet: A deep convolutional encoder-decoder architecture for image segmentation
# IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017
# Available from: https://arxiv.org/abs/1511.00561

class SegNet(nn.Module):
    """SegNet model.
    Args:
        num_classes (int): number of classes to segment
        channels (int): number of input channels
        drop_rate (float): dropout rate
    """
    def __init__(self, dims, num_classes, channels=3, img_channels = 3, drop_rate=0.5):
        super(SegNet, self).__init__()

        self.dims = dims
        self.img_channels = img_channels
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        if channels <= 3:
            filter_config=(64, 128, 256, 512, 512)
        else:
            filter_config=(channels, channels*2, channels*4, channels*8, channels*8)
        encoder_n_layers = (2, 2, 3, 3, 3)
        encoder_filter_config = (channels,) + filter_config
        decoder_n_layers = (3, 3, 3, 2, 1)
        decoder_filter_config = filter_config[::-1] + (filter_config[0],)

        for i in range(5):
            self.encoders.append(_Encoder(dims, encoder_filter_config[i],
                                          encoder_filter_config[i + 1],
                                          encoder_n_layers[i], drop_rate))

            self.decoders.append(_Decoder(dims, decoder_filter_config[i],
                                          decoder_filter_config[i + 1],
                                          decoder_n_layers[i], drop_rate))

        if self.dims == 2:
            self.classifier = nn.Conv2d(filter_config[0], num_classes, 3, 1, 1)
        else: # self.dims == 3:
            classifier = [nn.Conv2d(filter_config[0]*self.img_channels, self.img_channels, 3, 1, 1)]
            classifier.append(nn.Conv2d(self.img_channels, num_classes, 3, 1, 1))
            self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        indices = []
        unpool_sizes = []
        feat = x

        for i in range(5):
            (feat, ind), size = self.encoders[i](feat)
            indices.append(ind)
            unpool_sizes.append(size)

        for i in range(5):
            feat = self.decoders[i](feat, indices[4 - i], unpool_sizes[4 - i])

        if self.dims == 3:
            x = torch.flatten(feat,1,2)
        x = self.classifier(x)
        return x


class _Encoder(nn.Module):
    def __init__(self, dims, n_in_feat, n_out_feat, n_blocks=2, drop_rate=0.5,
                normalization = 'BatchNorm', activation = nn.ReLU(inplace=True)):
        super(_Encoder, self).__init__()

        self.dims = dims
        layers = []
        for layer in range(n_blocks):
            layers += [BasicConv(self.dims, n_in_feat if layer == 0 else n_out_feat, 
                            n_out_feat, kernel_size = 3, stride = 1, padding = 1, 
                            normalization = normalization,
                            activation = activation, act_first=False)]
        if n_blocks > 2:
                layers += [nn.Dropout(drop_rate)]

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        output = self.features(x)
        if self.dims == 2:
            return F.max_pool2d(output, 2, 2, return_indices=True), output.size()
        else: #self.dims == 3:
            return F.max_pool3d(output, 2, 2, return_indices=True), output.size()


class _Decoder(nn.Module):
    def __init__(self, dims, n_in_feat, n_out_feat, n_blocks=2, drop_rate=0.5,
                normalization = 'BatchNorm', activation = nn.ReLU(inplace=True)):
        super(_Decoder, self).__init__()

        self.dims = dims
        layers = []
        for layer in range(n_blocks):
            layers += [BasicConv(self.dims, n_in_feat if layer == 0 else n_out_feat, 
                            n_out_feat, kernel_size = 3, stride = 1, padding = 1, 
                            normalization = normalization,
                            activation = activation, act_first=False)]
        if n_blocks > 2:
                layers += [nn.Dropout(drop_rate)]

        self.features = nn.Sequential(*layers)

    def forward(self, x, indices, size):
        if self.dims == 2:
            unpooled = F.max_unpool2d(x, indices, 2, 2, 0, size)
        else: # self.dims == 3:
            unpooled = F.max_unpool3d(x, indices, 2, 2, 0, size)
        return self.features(unpooled)
        