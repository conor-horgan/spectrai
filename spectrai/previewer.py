import os
import sys
import numpy as np
module_path = os.path.abspath(os.path.join('../../spectrai'))
if module_path not in sys.path:
    sys.path.append(module_path)
import spectrai
import spectrai.initialise as initialise

def preview_preprocessing(Task_Options, Network_Hyperparameters, Training_Hyperparameters, Preprocessing, Data_Augmentation, DataManager_Options):
    """Returns preprocessed and augmented input and target data.

    For details on all input parameter dictionaries, see: 
    /spectrai/spectrai/configs/*.yml

    Arguments:
        Task_Options: dictionary of task options
        Network_Hyperparameters: dictionary of network hyperparameters
        Training_Hyperparameters: dictionary of training hyperparameters
        Preprocessing: dictionary of preprocessing options
        Data_Augmentation: dictionary of data augmentation options
        DataManager_Options: dictionary of data manager options
    
    Returns:
        output: dictionary containing preprocessed and augmented
            input and target data
    """
    preview_loader, _ , _ = initialise.initialise_dataset(Task_Options, Network_Hyperparameters, Training_Hyperparameters, Preprocessing, Data_Augmentation, DataManager_Options, False)
    preview_batch = next(iter(preview_loader))
    x = preview_batch['input']
    y = preview_batch['target']['data']
    x = np.squeeze(x[0,...].numpy())
    if Task_Options['task'] != 'Classification':
        y = np.squeeze(y[0,...].numpy())
    else:
        y = np.squeeze(y[0].numpy())

    x = np.moveaxis(x, 0, -1)
    x = np.ascontiguousarray(x)
    if Task_Options['task'] == 'Segmentation':
        y = np.ascontiguousarray(y)
    elif Task_Options['task'] != 'Classification':
        y = np.moveaxis(y, 0, -1)
        y = np.ascontiguousarray(y)

    output = {'input': x, 'target': y}
    return output

def preview_progress(Task_Options, net, device, preview_loader):
    """Returns preprocessed and augmented input and target data.

    For details on all input parameter dictionaries, see: 
    /spectrai/spectrai/configs/*.yml

    Arguments:
        Task_Options: dictionary of task options
        net: PyTorch neural network
        device: device used fr neural network computation
        preview_loader: PyTorch dataloader
    
    Returns:
        output: dictionary containing input, target, and
            neural network output data
    """
    preview_batch = next(iter(preview_loader))
    inputs = preview_batch['input']
    inputs = inputs.float()
    inputs = inputs.to(device)
    target = preview_batch['target']['data']
    if Task_Options['task'] == 'Segmentation' or Task_Options['task'] == 'Classification':
        target = target.long()
    else:
        target = target.float()
    #target = target.to(device)

    output = net(inputs)

    x = inputs.cpu().detach().numpy()
    y = output.cpu().detach().numpy()
    target = target.numpy()

    x = np.squeeze(x[0,...])
    x = np.moveaxis(x, 0, -1)
    x = np.ascontiguousarray(x)

    if Task_Options['task'] == 'Classification':
        y = np.squeeze(y[0])
        y = np.argmax(y)
        target = np.squeeze(target[0])
    elif Task_Options['task'] == 'Segmentation':
        y = np.squeeze(y[0,...])
        y = np.argmax(y,axis=0)
        y = np.ascontiguousarray(y)
        target = np.squeeze(target[0,...])
        target = np.ascontiguousarray(target)
    else:
        y = np.squeeze(y[0,...])
        y = np.moveaxis(y, 0, -1)
        y = np.ascontiguousarray(y)
        target = np.squeeze(target[0,...])
        target = np.moveaxis(target, 0, -1)
        target = np.ascontiguousarray(target)

    output = {'input': x, 'output': y, 'target': target}

    return output