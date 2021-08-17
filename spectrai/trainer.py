import os
import sys
import random
import datetime
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
module_path = os.path.abspath(os.path.join('../../spectrai'))
if module_path not in sys.path:
    sys.path.append(module_path)
import spectrai
import spectrai.initialise as initialise
import spectrai.utils.utilities as utilities
import spectrai.previewer as previewer

def train_epoch(Training_Options, Task_Options, Network_Hyperparameters, Training_Hyperparameters, Preprocessing, 
                Data_Augmentation, DataManager_Options, net_state_dict, optimizer_state_dict, scheduler_state_dict,
                epochs, results_preview = 0, current_epoch = 1, max_epochs = 10, save_frequency = 0, verbose = False):
    """Initialises and trains a neural network given user input parameters.

    For details on all input parameter dictionaries, see: 
    /spectrai/spectrai/configs/*.yml

    Arguments:
        Training_Options: dictionary of training options
        Task_Options: dictionary of task options
        Network_Hyperparameters: dictionary of network hyperparameters
        Training_Hyperparameters: dictionary of training hyperparameters
        Preprocessing: dictionary of preprocessing options
        Data_Augmentation: dictionary of data augmentation options
        DataManager_Options: dictionary of data manager options
        net_state_dict: PyTorch model state_dict or 'None' (str)
        optimizer_state_dict: PyTorch optimizer state_dict or 'None' (str)
        scheduler_state_dict: PyTorch scheduler state_dict or 'None' (str)
        epochs: number of epochs to train
        results_preview: flag indicating whether to output preview of
            results during training (MATLAB GUI only)
        current_epoch: current epoch counter for MATLAB loss evaluation
            (MATLAB GUI only)
        max_epochs: maximum number of epochs to train, set equal to 
            epochs unless using MATLAB GUI
        save_frequency: frequency (epochs) with which to save model
            overwrites by default
    
    Returns:
        output: dictionary containing the following fields
            network: network state_dict
            optimizer: optimizer state_dict
            scheduler: scheduler state_dict
            criterion: PyTorch criterion
            classes: classification/segmentation classes else ignored
            train_metrics: dictionary of training metrics
            val_metrics: dictionary of validation metrics
            test_metrics: dictionary of test metrics
            previewer: previewer output dictionary if results_preview == 1
    """
    utilities.check_inputs(Training_Options, Task_Options, Network_Hyperparameters, 
                        Training_Hyperparameters, Preprocessing, Data_Augmentation, 
                        DataManager_Options)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if DataManager_Options['seed'] != 'None':
        random.seed(int(DataManager_Options['seed']))
        torch.manual_seed(int(DataManager_Options['seed']))
        cudnn.deterministic = True

    net = initialise.initialise_network(Training_Options, Task_Options, Network_Hyperparameters, 
                                        Training_Hyperparameters, DataManager_Options, net_state_dict)
    net.to(device)

    train_loader, val_loader, test_loader = initialise.initialise_dataset(Task_Options, Network_Hyperparameters, 
                                                                        Training_Hyperparameters, Preprocessing, 
                                                                        Data_Augmentation, DataManager_Options)

    criterion, optimizer, scheduler = initialise.initialise_training_hyperparameters(Training_Hyperparameters, net, train_loader)

    if optimizer_state_dict != 'None':
        optimizer.load_state_dict(optimizer_state_dict)

    if scheduler is not None and scheduler_state_dict != 'None':
        scheduler.load_state_dict(scheduler_state_dict)

    date = datetime.datetime.now().strftime("%Y_%m_%d")
    log_dir = "tensorboard/{}_{}_{}".format(date, Task_Options['task'], Network_Hyperparameters['network'])
    writer = SummaryWriter(log_dir = log_dir)

    for epoch in range(int(epochs)):
        if train_loader is not None:
            train_metrics = train(Task_Options, Training_Hyperparameters, Data_Augmentation, net, criterion, optimizer, scheduler, train_loader, device)
            if int(results_preview) and val_loader is None:
                previewer_output = previewer.preview_progress(Task_Options, net, device, train_loader)
            if verbose:
                print('Epoch:', str(epoch), 'Train:', train_metrics)
                for key in train_metrics.keys():
                    writer.add_scalar('Train/'+key, train_metrics[key], epoch)
        else:
            train_metrics = None

        if val_loader is not None:
            val_metrics = validate(Task_Options, net, criterion, val_loader, device)
            if int(results_preview):
                previewer_output = previewer.preview_progress(Task_Options, net, device, val_loader)
            if verbose:
                print('Epoch:', str(epoch), 'Validation:', val_metrics)
                for key in val_metrics.keys():
                    writer.add_scalar('Validation/'+key, val_metrics[key], epoch)
        else:
            val_metrics = None

        if test_loader is not None:
            test_metrics = validate(Task_Options, net, criterion, test_loader, device)
        else:
            test_metrics = None

        if Training_Hyperparameters['scheduler'] == "Step" or Training_Hyperparameters['scheduler'] == "Multiplicative":
            scheduler.step()
        elif Training_Hyperparameters['scheduler'] == "ReduceOnPlateau":
            if val_metrics is not None:
                scheduler.step(val_metrics['loss'])
            elif train_metrics is not None:
                scheduler.step(train_metrics['loss'])
            else:
                raise ValueError('ReduceOnPlateau LR should only be used when training, ideally with a validation dataset')

        current_epoch += 1
        if (save_frequency != 0 and current_epoch % save_frequency == 0) or current_epoch == max_epochs:
            save_output(Network_Hyperparameters, Task_Options, Training_Hyperparameters, DataManager_Options, 
                        net.state_dict(), optimizer.state_dict(), scheduler_state_dict, criterion, train_metrics, val_metrics, test_metrics)
    
    classes = int(Task_Options['classes'])
    output = {'network':net.state_dict(), 'optimizer':optimizer.state_dict(), 'scheduler':scheduler_state_dict, 'criterion':criterion, 
    'classes': classes, 'train_metrics':train_metrics, 'val_metrics':val_metrics, 'test_metrics':test_metrics}
    
    if int(results_preview):
        output['preview'] = previewer_output

    return output

def save_output(Network_Hyperparameters, Task_Options, Training_Hyperparameters, DataManager_Options, net, optimizer, 
                scheduler, criterion, train_metrics, val_metrics, test_metrics):
    """Saves trained neural network state_dict and associated parameters.

    For details on all input parameter dictionaries, see: 
    /spectrai/spectrai/configs/*.yml

    Arguments:
        Network_Hyperparameters: dictionary of network hyperparameters
        Task_Options: dictionary of task options
        Training_Hyperparameters: dictionary of training hyperparameters
        DataManager_Options: dictionary of data manager options
        net: PyTorch model state_dict or 'None' (str)
        optimizer: PyTorch optimizer state_dict or 'None' (str)
        scheduler: PyTorch scheduler state_dict or 'None' (str)
        criterion: PyTorch criterion
        train_metrics: dictionary of training metrics
        val_metrics: dictionary of validation metrics
        test_metrics: dictionary of test metrics
    
    Returns:
        output: dictionary containing the following fields
            network: network state_dict
            optimizer: optimizer state_dict
            scheduler: scheduler state_dict
            criterion: criterion
            train_metrics: dictionary of training metrics
            val_metrics: dictionary of validation metrics
            test_metrics: dictionary of test metrics
            achitecture: network achitecture [UNet/ResUNet/ResNet/RCAN]
            task: deep learning task [Calibration/Classification/Denoising/Segmentation/Super-Resolution]
            channels: spectrum length (int)
            normalization: normalization layer [None/BatchNorm/LayerNorm/InstanceNorm/GroupNorm]
            activation: activation function [ReLU/LeakyReLU/PReLU]
            classes: classification/segmentation classes else ignored
            input_image_size: input image size (int)
            target_image_size: target image size (int)
            data_format: data format ['Image: H, W, C'/'Image: C, H, W'/Spectra]
    """
    architecture = Network_Hyperparameters['network']
    task = Task_Options['task']
    channels = int(Training_Hyperparameters['spectrum_length'])
    normalization = Network_Hyperparameters['normalization']
    activation = Network_Hyperparameters['activation']
    classes = int(Task_Options['classes'])
    input_image_size = int(Training_Hyperparameters['input_image_size'])
    target_image_size = int(Training_Hyperparameters['target_image_size'])
    data_format = DataManager_Options['data_format']

    output = {'network':net, 'optimizer':optimizer, 'scheduler':scheduler, 'criterion':criterion, 
        'train_metrics':train_metrics, 'val_metrics':val_metrics, 'test_metrics':test_metrics,
        'architecture':architecture, 'task':task, 'channels':channels, 'normalization':normalization, 
        'activation':activation, 'classes':classes, 'input_image_size':input_image_size, 
        'target_image_size':target_image_size, 'data_format':data_format}

    date = datetime.datetime.now().strftime("%Y_%m_%d")
    save_name = date + '_' + task + '_' + architecture + '.pt'

    dir_path = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(dir_path, 'saved_models')):
        os.makedirs(os.path.join(dir_path, 'saved_models'))
    torch.save(output, os.path.join(dir_path, 'saved_models', save_name))

def load_pretrained(pretrained_network):
    """Loads parameters of neural network pretrained using spectrai.

    Arguments:
        pretrained_network: dictionary containing the following fields
            network: network state_dict
            optimizer: optimizer state_dict
            scheduler: scheduler state_dict
            criterion: criterion
            train_metrics: dictionary of training metrics
            val_metrics: dictionary of validation metrics
            test_metrics: dictionary of test metrics
            achitecture: network achitecture [UNet/ResUNet/ResNet/RCAN]
            task: deep learning task [Calibration/Classification/Denoising/Segmentation/Super-Resolution]
            channels: spectrum length (int)
            normalization: normalization layer [None/BatchNorm/LayerNorm/InstanceNorm/GroupNorm]
            activation: activation function [ReLU/LeakyReLU/PReLU]
            classes: classification/segmentation classes else ignored
            input_image_size: input image size (int)
            target_image_size: target image size (int)
            data_format: data format ['Image: H, W, C'/'Image: C, H, W'/Spectra]
    Returns:
        output: dictionary containing the following fields
            achitecture: network achitecture [UNet/ResUNet/ResNet/RCAN]
            task: deep learning task [Calibration/Classification/Denoising/Segmentation/Super-Resolution]
            channels: spectrum length (int)
            normalization: normalization layer [None/BatchNorm/LayerNorm/InstanceNorm/GroupNorm]
            activation: activation function [ReLU/LeakyReLU/PReLU]
            classes: classification/segmentation classes else ignored
            input_image_size: input image size (int)
            target_image_size: target image size (int)
            data_format: data format ['Image: H, W, C'/'Image: C, H, W'/Spectra]
    """
    input_data = torch.load(pretrained_network)
    architecture = input_data['architecture']
    task = input_data['task']
    channels = input_data['channels']
    normalization = input_data['normalization']
    activation = input_data['activation']
    classes = input_data['classes']
    input_image_size = input_data['input_image_size']
    target_image_size = input_data['target_image_size']
    data_format = input_data['data_format']

    output = {'architecture':architecture, 'task':task, 'channels':channels,
            'normalization':normalization, 'activation':activation, 'classes':classes, 
            'input_image_size':input_image_size, 'target_image_size':target_image_size, 
            'data_format':data_format}

    return output

def train(Task_Options, Training_Hyperparameters, Data_Augmentation, net, criterion, optimizer, scheduler, dataloader, device):
    """Performs a train step for a given neural network.

    For details on all input parameter dictionaries, see: 
    /spectrai/spectrai/configs/*.yml

    Arguments:
        Task_Options: dictionary of task options
        Training_Hyperparameters: dictionary of training hyperparameters
        Data_Augmentation: dictionary of data augmentation options
        net: PyTorch model state_dict or 'None' (str)
        criterion: PyTorch criterion
        optimizer: PyTorch optimizer state_dict or 'None' (str)
        scheduler: PyTorch scheduler state_dict or 'None' (str)
        dataloader: PyTorch dataloader
        device: device used fr neural network computation
    
    Returns:
        metrics: dictionary containing various loss and performance metrics
    """
    batch_time = utilities.AverageMeter('Time', ':6.3f')
    losses = utilities.AverageMeter('Loss', ':.4e')
    if Task_Options['task'] == 'Super-Resolution':
        psnr = utilities.AverageMeter('PSNR', ':.4f')
        ssim = utilities.AverageMeter('SSIM', ':.4f')
    elif Task_Options['task'] == 'Classification':
        acc = utilities.AverageMeter('Accuracy', ':.4f')

    end = time.time()
    for i, sample in enumerate(dataloader):
        inputs = sample['input']
        inputs = inputs.float()
        inputs = inputs.to(device)
        target = sample['target']['data']
        if Task_Options['task'] == 'Segmentation' or Task_Options['task'] == 'Classification':
            target = target.long()
        else:
            target = target.float()
        target = target.to(device)

        if Data_Augmentation['mixup']:
            inputs, target_a, target_b, lam = utilities.mixup_data(inputs, target, alpha = 0.2)
            inputs, target_a, target_b = map(Variable, (inputs, target_a, target_b))
            output = net(inputs)
            loss = utilities.mixup_criterion(criterion, output, target_a, target_b, lam)
        else:
            output = net(inputs)
            loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if Training_Hyperparameters['scheduler'] == "Cyclic" or Training_Hyperparameters['scheduler'] == "OneCycle":
            scheduler.step()        

        losses.update(loss.item(), inputs.size(0)) 

        batch_time.update(time.time() - end)
        end = time.time()

        metrics = {'loss': losses.avg}

        if Task_Options['task'] == 'Super-Resolution':
            psnr_batch = utilities.calc_psnr(output, target)
            psnr.update(psnr_batch, inputs.size(0))
            
            ssim_batch = utilities.calc_ssim(output, target)
            ssim.update(ssim_batch, inputs.size(0))

            metrics['psnr'] = psnr.avg
            metrics['ssim'] = ssim.avg
        elif Task_Options['task'] == 'Classification':
            # TO DO: implement for mixup
            _, preds = torch.max(output, 1)
            acc_batch = torch.sum(preds == target.data).item()/inputs.size(0)
            acc.update(acc_batch, inputs.size(0))

            metrics['accuracy'] = acc.avg

    return metrics

def validate(Task_Options, net, criterion, dataloader, device):
    """Performs a train step for a given neural network.

    For details on all input parameter dictionaries, see: 
    /spectrai/spectrai/configs/*.yml

    Arguments:
        Task_Options: dictionary of task options
        Data_Augmentation: dictionary of data augmentation options
        net: PyTorch model state_dict or 'None' (str)
        criterion: PyTorch criterion
        optimizer: PyTorch optimizer state_dict or 'None' (str)
        scheduler: PyTorch scheduler state_dict or 'None' (str)
        dataloader: PyTorch dataloader
        device: device used fr neural network computation
    
    Returns:
        metrics: dictionary containing various loss and performance metrics
    """
    batch_time = utilities.AverageMeter('Time', ':6.3f')
    losses = utilities.AverageMeter('Loss', ':.4e')
    if Task_Options['task'] == 'Super-Resolution':
        psnr = utilities.AverageMeter('PSNR', ':.4f')
        ssim = utilities.AverageMeter('SSIM', ':.4f')
    elif Task_Options['task'] == 'Classification':
        acc = utilities.AverageMeter('Accuracy', ':.4f')

    net.eval()

    with torch.no_grad():
        end = time.time()
        for i, sample in enumerate(dataloader):
            inputs = sample['input']
            inputs = inputs.float()
            inputs = inputs.to(device)
            target = sample['target']['data']
            if Task_Options['task'] == 'Segmentation' or Task_Options['task'] == 'Classification':
                target = target.long()
            else:
                target = target.float()
            target = target.to(device)

            output = net(inputs)    

            loss = criterion(output, target)
            losses.update(loss.item(), inputs.size(0))               
            
            batch_time.update(time.time() - end)
            end = time.time()

            metrics = {'loss': losses.avg}

            if Task_Options['task'] == 'Super-Resolution':
                psnr_batch = utilities.calc_psnr(output, target)
                psnr.update(psnr_batch, inputs.size(0))
                
                ssim_batch = utilities.calc_ssim(output, target)
                ssim.update(ssim_batch, inputs.size(0))

                metrics['psnr'] = psnr.avg
                metrics['ssim'] = ssim.avg
            elif Task_Options['task'] == 'Classification':
                _, preds = torch.max(output, 1)
                acc_batch = torch.sum(preds == target.data).item()/inputs.size(0)
                acc.update(acc_batch, inputs.size(0))

                metrics['accuracy'] = acc.avg

    return metrics

def evaluate_pretrained(Training_Options, Task_Options, Network_Hyperparameters, Training_Hyperparameters, Preprocessing, 
                Data_Augmentation, DataManager_Options, net_state_dict, results_preview = 0):
    """Performs a evaluation for a given pretrained neural network.

    For details on all input parameter dictionaries, see: 
    /spectrai/spectrai/configs/*.yml

    Arguments:
        Training_Options: dictionary of training options
        Task_Options: dictionary of task options
        Network_Hyperparameters: dictionary of network hyperparameters
        Training_Hyperparameters: dictionary of training hyperparameters
        Preprocessing: dictionary of preprocessing options
        Data_Augmentation: dictionary of data augmentation options
        DataManager_Options: dictionary of data manager options
        net_state_dict: PyTorch model state_dict or 'None' (str)
        results_preview: flag indicating whether to output preview of
            results during training (MATLAB GUI only)
    
    Returns:
        output: dictionary containing the following fields
            test_metrics: dictionary containing various loss and performance metrics
            preview: previewer output dictionary if results_preview == 1
    """
    utilities.check_inputs(Training_Options, Task_Options, Network_Hyperparameters, 
                        Training_Hyperparameters, Preprocessing, Data_Augmentation, 
                        DataManager_Options)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if DataManager_Options['seed'] != 'None':
        random.seed(int(DataManager_Options['seed']))
        torch.manual_seed(int(DataManager_Options['seed']))
        cudnn.deterministic = True

    net = initialise.initialise_network(Training_Options, Task_Options, Network_Hyperparameters, Training_Hyperparameters, 
                                        DataManager_Options, net_state_dict)
    net.to(device)

    _, _, test_loader = initialise.initialise_dataset(Task_Options, Network_Hyperparameters, Training_Hyperparameters, Preprocessing, 
                                                                        Data_Augmentation, DataManager_Options)

    if test_loader is not None:
        criterion, _, _ = initialise.initialise_training_hyperparameters(Training_Hyperparameters, net, test_loader)
    else:
        criterion, _, _ = initialise.initialise_training_hyperparameters(Training_Hyperparameters, net, None)

    if test_loader is not None:
        test_metrics = validate(Task_Options, net, criterion, test_loader, device)
        if int(results_preview):
            previewer_output = previewer.preview_progress(Task_Options, net, device, test_loader)
    else:
        test_metrics = None

    output = {'test_metrics': test_metrics}

    if int(results_preview):
        output['preview'] = previewer_output

    return output

def apply_pretrained(Training_Options, Task_Options, Network_Hyperparameters, Training_Hyperparameters, Preprocessing, 
                Data_Augmentation, DataManager_Options, net_state_dict, results_preview = 0, apply = 1):
    """Prepares application of pretrained neural network to dataset with no target data.

    For details on all input parameter dictionaries, see: 
    /spectrai/spectrai/configs/*.yml

    Arguments:
        Training_Options: dictionary of training options
        Task_Options: dictionary of task options
        Network_Hyperparameters: dictionary of network hyperparameters
        Training_Hyperparameters: dictionary of training hyperparameters
        Preprocessing: dictionary of preprocessing options
        Data_Augmentation: dictionary of data augmentation options
        DataManager_Options: dictionary of data manager options
        net_state_dict: PyTorch model state_dict or 'None' (str)
        results_preview: flag indicating whether to output preview of
            results during training (MATLAB GUI only)
        apply: flag indicating that target data is unavailable for
            training/evaluation
    
    Returns:
        output: dictionary containing the following fields or None
            preview: previewer output dictionary if results_preview == 1
    """
    utilities.check_inputs(Training_Options, Task_Options, Network_Hyperparameters, 
                        Training_Hyperparameters, Preprocessing, Data_Augmentation, 
                        DataManager_Options)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = initialise.initialise_network(Training_Options, Task_Options, Network_Hyperparameters, Training_Hyperparameters, 
                                        DataManager_Options, net_state_dict)
    net.to(device)

    _, _, test_loader = initialise.initialise_dataset(Task_Options, Network_Hyperparameters, Training_Hyperparameters, Preprocessing, 
                                                    Data_Augmentation, DataManager_Options, apply)
    
    if test_loader is not None:
        apply_net(Task_Options, Network_Hyperparameters, net, test_loader, device)
        if int(results_preview):
            previewer_output = previewer.preview_progress(Task_Options, net, device, test_loader)

    if int(results_preview):
        output = {'preview': previewer_output}
    else:
        output = None

    return output

def apply_net(Task_Options, Network_Hyperparameters, net, dataloader, device):
    """Applies pretrained neural network to dataset with no target data.

    Neural network outputs for each sample in the dataset will be saved
    in /spectrai/spectrai/apply_outputs/

    For details on all input parameter dictionaries, see: 
    /spectrai/spectrai/configs/*.yml

    Arguments:
        Task_Options: dictionary of task options
        Network_Hyperparameters: dictionary of network hyperparameters
        net: PyTorch model state_dict or 'None' (str)
        dataloader: PyTorch dataloader
        device: device used fr neural network computation
    """
    architecture = Network_Hyperparameters['network']
    task = Task_Options['task']
    dir_path = os.path.dirname(os.path.realpath(__file__))
    date = datetime.datetime.now().strftime("%Y_%m_%d")
    save_name = date + '_' + task + '_' + architecture

    net.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            names = data['name']
            inputs = data['input']
            inputs = inputs.float()
            inputs = inputs.to(device)
            outputs = net(inputs)
            
            net_out = outputs.detach().cpu().numpy()

            for j in range(net_out.shape[0]):
                output_data = np.squeeze(net_out[j])
                file_name = names[j] + '_output.npy'
                if not os.path.exists(os.path.join(dir_path, 'apply_outputs', save_name)):
                    os.makedirs(os.path.join(dir_path, 'apply_outputs', save_name))
                np.save(os.path.join(dir_path, 'apply_outputs', save_name, file_name), output_data)
    return