import math
import torch
from torch import nn
from spectrai.networks.layers import BasicConv
from spectrai.networks.nets.rcan import Hyperspectral_RCAN
from spectrai.networks.nets.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from spectrai.networks.nets.unet import UNet
from spectrai.networks.nets.efficientnet import EfficientNet
from spectrai.networks.nets.segnet import SegNet
from spectrai.networks.nets.densenet import densenet121, densenet161, densenet169, densenet201

def setup_network(Task_Options, Network_Hyperparameters, Training_Hyperparameters, DataManager_Options):
    """Initialises a neural network given user input parameters.

    For details on all input parameter dictionaries, see: 
    /spectrai/spectrai/configs/*.yml

    Arguments:
        Task_Options: dictionary of task options
        Network_Hyperparameters: dictionary of network hyperparameters
        Training_Hyperparameters: dictionary of training hyperparameters
        DataManager_Options: dictionary of data manager options
    
    Returns:
        net: PyTorch neural network
    """
    task = Task_Options['task']
    channels = int(Training_Hyperparameters['spectrum_length'])
    image_size = int(Training_Hyperparameters['target_image_size'])
    normalization = Network_Hyperparameters['normalization']

    if task == 'Classification' or task == 'Segmentation':
        num_classes = int(Task_Options['classes'])
    else: 
        num_classes = 0

    if task == 'Super-Resolution':
        scale = int(Training_Hyperparameters['target_image_size']) // int(Training_Hyperparameters['input_image_size'])
    else:
        scale = None

    if Network_Hyperparameters['activation'] == 'PReLU':
        activation = nn.PReLU()
    elif Network_Hyperparameters['activation'] == 'LeakyReLU':
        activation = nn.LeakyReLU()
    else: # Network_Hyperparameters['activation'] == 'ReLU':
        activation = nn.ReLU()

    if Network_Hyperparameters['dimension'] == '2D':
        dims = 2
    else: # Network_Hyperparameters['dimension'] == '3D':
        dims = 3

    # ------------UNet/ResUNet-----------
    if Network_Hyperparameters['network'] == 'UNet' or Network_Hyperparameters['network'] == 'ResUNet':
        res = True if Network_Hyperparameters['network'] == 'ResUNet' else False
        if DataManager_Options['data_format'] == 'Spectra':
            if task not in ['Calibration', 'Classification', 'Denoising']:
                raise NotImplementedError('1D UNet/ResUNet architecture (for spectra) only implemented for calibration, classification, and denoising')
            else:
                net = UNet(dims = 1, channels = 1, num_classes = num_classes, n_blocks = 3, normalization = normalization, activation = activation, task = task, res = res).float()
        else: # Hyperspectral Image Data
            if dims == 2:
                net = UNet(dims = dims, channels = channels, num_classes = num_classes, n_blocks = 3, normalization = normalization, activation = activation,  task = task, scale = scale, res = res).float()
            elif dims == 3:
                if task == 'Super-Resolution':
                    raise NotImplementedError('UNet/ResUNet architecture not implemented for 3D super-resolution')
                else:
                    net = UNet(dims = dims, channels = 1, img_channels = channels, num_classes = num_classes, n_blocks = 3, normalization = normalization, activation = activation,  task = task, scale = scale, res = res).float()

    # --------------ResNet--------------
    elif Network_Hyperparameters['network'] == 'ResNet':
        if task == 'Classification':
            if DataManager_Options['data_format'] == 'Spectra':
                net = ResNet18(dims = 1, img_channels = channels, res_channels = channels, num_classes = num_classes, normalization = normalization, activation = activation).float()
            else: # Hyperspectral Image Data
                if dims == 2:
                    net = ResNet18(dims = dims, img_channels = channels, res_channels = channels, num_classes = num_classes, normalization = normalization, activation = activation).float()
                elif dims == 3:
                    net = ResNet18(dims = dims, img_channels = 1, res_channels = 64, num_classes = num_classes, normalization = normalization, activation = activation).float()
        else:
            raise NotImplementedError('ResNet architecture has only been implemented for classification')

    # ---------------RCAN---------------
    elif Network_Hyperparameters['network'] == 'RCAN':
        if DataManager_Options['data_format'] == 'Spectra':
            raise NotImplementedError('RCAN architecture not implemented for 1D (spectral) data')
        else: # Hyperspectral Image Data
            if task == 'Super-Resolution' and dims == 2:
                net = Hyperspectral_RCAN(spectrum_length = channels, scale = scale, activation = activation).float()
            else:
                raise NotImplementedError('RCAN architecture only implemented for 2D super-resolution')

    # -----------EfficientNet------------
    elif Network_Hyperparameters['network'] == 'EfficientNet':
        if DataManager_Options['data_format'] == 'Spectra':
            raise NotImplementedError('EfficientNet architecture not implemented for 1D (spectral) data')
        else: # Hyperspectral Image Data
            if task == 'Classification' and dims == 2:
                override_params = {'image_size': image_size, 'num_classes': num_classes}
                net = EfficientNet.from_name('efficientnet-b0', channels, **override_params).float()
            else:
                raise NotImplementedError('EfficientNet architecture only implemented for 2D classification')

    # -------------SegNet---------------
    elif Network_Hyperparameters['network'] == 'SegNet':
        if DataManager_Options['data_format'] == 'Spectra':
            raise NotImplementedError('SegNet architecture not implemented for 1D (spectral) data')
        else: # Hyperspectral Image Data
            if task == 'Segmentation':
                if dims == 2:
                    net = SegNet(dims = dims, num_classes = num_classes, channels = channels).float()
                elif dims == 3:
                    net = SegNet(dims = dims, num_classes = num_classes, channels = 1, img_channels = channels).float()
            else:
                raise NotImplementedError('SegNet architecture only implemented for segmentation')

    # -------------DenseNet---------------
    elif Network_Hyperparameters['network'] == 'DenseNet':
        if task == 'Classification':
            if DataManager_Options['data_format'] == 'Spectra':
                net = densenet121(dims = 1, channels = 1, num_classes = num_classes, normalization = normalization, activation = activation).float()
            else: # Hyperspectral Image Data
                if dims == 2:
                    net = densenet121(dims = dims, channels = channels, num_classes = num_classes, normalization = normalization, activation = activation).float()
                elif dims == 3:
                    net = densenet121(dims = dims, channels = 1, num_classes = num_classes, normalization = normalization, activation = activation).float()
        else:
            raise NotImplementedError('ResNet architecture has only been implemented for classification')

    else:
        raise ValueError('%s is not a valid network' %Network_Hyperparameters['network'])

    return net

def edit_network(net, Task_Options, Network_Hyperparameters, Training_Hyperparameters, DataManager_Options):
    """Edits neural network to match desired number of output classes.

    For details on all input parameter dictionaries, see: 
    /spectrai/spectrai/configs/*.yml

    Arguments:
        net: PyTorch neural network
        Task_Options: dictionary of task options
        Network_Hyperparameters: dictionary of network hyperparameters
        Training_Hyperparameters: dictionary of training hyperparameters
        DataManager_Options: dictionary of data manager options
    
    Returns:
        net: PyTorch neural network
    """
    classes = int(Task_Options['classes'])
    channels = int(Training_Hyperparameters['spectrum_length'])
    if Network_Hyperparameters['network'] == 'UNet' or Network_Hyperparameters['network'] == 'ResUNet':
        if DataManager_Options['data_format'] == 'Spectra':
            net.tail = nn.Linear(1, classes)
        else:
            if Task_Options['task'] == 'Classification':
                net.tail = nn.Linear(channels, classes)
            elif Task_Options['task'] == 'Segmentation':
                if Network_Hyperparameters['dimension'] == '2D':
                    net.tail = BasicConv(2, net.filter_config[0], classes, kernel_size = 1, normalization = None, activation = None)
                else:
                    net.tail = BasicConv(2, net.img_channels, classes, kernel_size = 1, normalization = None, activation = None)
    elif Network_Hyperparameters['network'] == 'ResNet':
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features, classes)
    elif Network_Hyperparameters['network'] == 'EfficientNet':
        in_features = net._fc.in_features
        net._fc = nn.Linear(in_features, classes)
    elif Network_Hyperparameters['network'] == 'SegNet':
        if Network_Hyperparameters['dimension'] == '2D':
            net.classifier = nn.Conv2d(channels, classes, kernel_size = 3, stride = 1, padding = 1)
        else:
            net.classifier[1] = nn.Conv2d(net.img_channels, classes, kernel_size = 3, stride = 1, padding = 1)
    return net
    