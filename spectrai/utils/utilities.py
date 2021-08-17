import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import scipy.signal
from skimage.metrics import structural_similarity as sk_ssim
import math
import yaml
import torch
from torch import nn

def check_inputs(Training_Options, Task_Options, Network_Hyperparameters, Training_Hyperparameters, 
                Preprocessing, Data_Augmentation, DataManager_Options):
    """Performs initial check of user input parameters."""
    # -----------------Task_Options-----------------
    if Task_Options['task'] not in ['Calibration', 'Classification', 'Denoising', 'Segmentation', 'Super-Resolution']:
        raise ValueError('%s is not a valid task' %Task_Options['task'])

    # ---------------Training_Options---------------
    if Training_Options['training_option'] not in ['Apply Pre-Trained Network', 'Train From Scratch', 'Transfer Learning']:
        raise ValueError('%s is not a valid training option' %Training_Options['training_option'])

    # -----------Network_Hyperparameters------------
    if Network_Hyperparameters['network'] not in ['UNet', 'ResUNet', 'ResNet', 'RCAN', 'EfficientNet', 'SegNet', 'DenseNet']:
        raise ValueError('%s is not a valid network' %Network_Hyperparameters['network'])

    if Network_Hyperparameters['dimension'] not in ['2D', '3D']:
        raise ValueError('%s is not a valid dimension' %Network_Hyperparameters['dimension'])

    if Network_Hyperparameters['activation'] not in ['ReLU', 'PReLU', 'LeakyReLU']:
        raise ValueError('%s is not a valid activation' %Network_Hyperparameters['activation'])

    if Network_Hyperparameters['normalization'] not in ['BatchNorm', 'GroupNorm', 'InstanceNorm', 'LayerNorm', 'None']:
        raise ValueError('%s is not a valid normalization' %Network_Hyperparameters['normalization'])

    # -----------Training_Hyperparameters------------
    if Training_Hyperparameters['optimizer'] not in ['Adam', 'Adagrad', 'SGD', 'RMSprop']:
        raise ValueError('%s is not a valid optimizer' %Training_Hyperparameters['optimizer'])

    if Training_Hyperparameters['scheduler'] not in ['Step', 'Multiplicative', 'Cyclic', 'OneCycle', 'ReduceOnPlateau', 'Constant']:
        raise ValueError('%s is not a valid scheduler' %Training_Hyperparameters['scheduler'])

    if Training_Hyperparameters['criterion'] not in ['L1', 'L2 / MSE', 'Cross Entropy', 'Binary Cross Entropy']:
        raise ValueError('%s is not a valid criterion' %Training_Hyperparameters['criterion'])

    # ----------------Preprocessing------------------
    if Preprocessing['background_subtraction'] not in ['None', 'Automatic Least Squares', '3rd Order Polynomial', '5th Order Polynomial', 'Minimum Value Offset']:
        raise ValueError('%s is not a valid background subtraction' %Preprocessing['background_subtraction'])

    if Preprocessing['data_normalization'] not in ['None', 'Max Value', 'Area Under The Curve']:
        raise ValueError('%s is not a valid data normalization' %Preprocessing['data_normalization'])

    # --------------DataManager_Options--------------
    if DataManager_Options['data_format'] not in ['Image: H, W, C', 'Image: C, H, W', 'Spectra']:
        raise ValueError('%s is not a valid data format' %DataManager_Options['data_format'])
    return

def get_config(config):
    """Loads a yaml configuration file."""
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def get_file_list(directory, extension):
    """Loads all files in given directory with given extension."""
    data_files = []
    for data_file in os.listdir(directory):
        if data_file.endswith(extension):
            data_files.append(os.path.join(directory, data_file))
    return data_files

def load_sample(path):
    """Loads an individual .mat or .npy file."""
    _, extension = os.path.splitext(path)

    if extension == '.mat':       
        output_data = scipy.io.loadmat(path)
        output_values = list(output_data.values())
        output_image = output_values[3]
    elif extension == '.npy':
        output_image = np.load(path)
    else:
        raise ValueError('Input file with extension %s is not a valid file type' %extension)
    return output_image

def load_mask(path):
    """Loads an individual image (mask) file."""
    mask_image = Image.open(path)
    mask = np.asarray(mask_image)
    mask.astype(np.float64)
    mask = mask - 1
    return mask

def prepare_dataset_directory(dataset_path):
    """Creates a dataset dataframe for a given directory.

    Arguments:
        dataset_path: directory containing all of the data for 
            a given dataset.
    
    Returns:
        dataset: a Pandas dataframe containing one columns;
            Data - paths to individual samples
    """
    data = []

    for file in os.listdir(dataset_path):
        data.append(os.path.join(dataset_path,file))

    dataset = {'Data':data} 
    dataset = pd.DataFrame(dataset)
    return dataset

def prepare_dataset_from_input(sample, sample_type):
    """Creates a dataset dataframe from a single input file.

    Arguments:
        sample: a single file containing either a spectral image,
            an image mask, or multiple spectra in an array
        sample_type: string indicating sample type as one of
            ['spectrum', 'hyperspectral_image', 'mask']
    
    Returns:
        dataset: a Pandas dataframe containing one column;
            Data - paths to individual samples
    """
    if sample_type == 'spectrum':
        spectra = load_sample(sample)
        dataset = pd.DataFrame(((x,) for x in spectra), columns=['Data'])
    if sample_type == 'hyperspectral_image':
        hyperspectral_image = load_sample(sample)
        dataset = pd.DataFrame({"Data": [hyperspectral_image]})
    if sample_type == 'mask':
        mask = load_mask(sample)
        dataset = pd.DataFrame({"Data": [mask]})
    return dataset

def prepare_classification_dataset(dataset_path, sample_type):
    """Creates a classification dataframe for a given directory.

    Arguments:
        dataset_path: directory containing folders corresponding to
            each class, containing all of the data for that class.
    
    Returns:
        dataset: a Pandas dataframe containing two columns;
            Data - paths to individual samples
            Labels - corresponding label for individual samples
    """
    data = []
    labels = []
    lb = LabelEncoder()

    for folder in os.listdir(dataset_path):
        if not os.path.isfile(os.path.join('.', folder)):
            if len(os.listdir(os.path.join(dataset_path,folder))) > 1:
                for sample in os.listdir(os.path.join(dataset_path,folder)):
                    data.append(os.path.join(dataset_path,folder,sample))
                    labels.append(folder)
            else:
                for sample in os.listdir(os.path.join(dataset_path,folder)):
                    sample_dataset = prepare_dataset_from_input(os.path.join(dataset_path,folder,sample), sample_type)
                    for i in range(len(sample_dataset)):
                        data.append(sample_dataset['Data'][i])
                        labels.append(folder)

    dataset = {'Data':data, 'Labels':labels} 
    dataset = pd.DataFrame(dataset)
    dataset['Encoded_Labels'] = lb.fit_transform(dataset['Labels'])
    return dataset

class AverageMeter(object):
    """Class to record mini-batch metric values during training.

    Arguments:
        val: metric value for mini-batch
        sum: sum of mini-batch metric values
        count: number of mini-batches
        avg: average metric value over mini-batches for epoch
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def calc_psnr(output, target):
    """Calculates the PSNR between two spectral images."""
    psnr = 0.
    mse = nn.MSELoss()(output, target)
    psnr = 10 * math.log10(1/mse)
    return psnr

def calc_ssim(output, target):
    """Calculates the SSIM between two spectral images."""
    ssim = 0.
    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    
    if output.ndim == 4:
        for i in range(output.shape[0]):
            output_i = np.squeeze(output[i,:,:,:])
            output_i = np.moveaxis(output_i, -1, 0)
            target_i = np.squeeze(target[i,:,:,:])
            target_i = np.moveaxis(target_i, -1, 0)
            batch_size = output.shape[0]
            ssim += sk_ssim(output_i, target_i, data_range = output_i.max() - target_i.max(), multichannel=True)
    else:
        output_i = np.squeeze(output)
        output_i = np.moveaxis(output_i, -1, 0)
        target_i = np.squeeze(target)
        target_i = np.moveaxis(target_i, -1, 0)
        batch_size = 1
        ssim += sk_ssim(output_i, target_i, data_range = output_i.max() - target_i.max(), multichannel=True)
        
    ssim = ssim / batch_size
    return ssim

def mixup_data(x, y, alpha=1.0):
    """Performs mixup for two samples.

    Arguments:
        x: input sample
        y: corresponding target/label
        alpha: alpha parameter for beta distribution

    Returns:
        mixed_x: mixuped sample composed of samples a and b
        y_a: label a
        y_b: label b
        lam: lambda value (from beta distribution) for mixup
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Calculates mixup loss value for given criterion."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)