import torch
from torch import nn
import torch.optim as optim
import copy
import spectrai.dataloader.dataloaders as dataloaders
import spectrai.networks.networks as networks

def initialise_network(Training_Options, Task_Options, Network_Hyperparameters, Training_Hyperparameters, DataManager_Options, net_state_dict):
    """Initialises a neural network given user input parameters.

    For details on all input parameter dictionaries, see: 
    /spectrai/spectrai/configs/*.yml

    Arguments:
        Training_Options: dictionary of training options
        Task_Options: dictionary of task options
        Network_Hyperparameters: dictionary of network hyperparameters
        Training_Hyperparameters: dictionary of training hyperparameters
        DataManager_Options: dictionary of data manager options
        net_state_dict: PyTorch model state_dict or 'None' (str)
    
    Returns:
        net: PyTorch neural network
    """
    if Training_Options['pretrained_network'] == 'None':
        net = networks.setup_network(Task_Options, Network_Hyperparameters, Training_Hyperparameters, DataManager_Options)
        if net_state_dict != 'None':
            net.load_state_dict(net_state_dict)
    else: #Training_Options['pretrained_network'] != 'None':
        if int(Training_Options['pretrained_classes']) == int(Task_Options['classes']):
            net = networks.setup_network(Task_Options, Network_Hyperparameters, Training_Hyperparameters, DataManager_Options)
            pretrained = torch.load(Training_Options['pretrained_network'])
            net.load_state_dict(pretrained['network'])
        else: #int(Training_Options['pretrained_classes']) != int(Task_Options['classes'])
            new_classes = Task_Options['classes']
            pretrained_classes = Training_Options['pretrained_classes']

            # Initialise network with pretrained_classes and load pretrained weights
            Task_Options['classes'] = pretrained_classes
            net = networks.setup_network(Task_Options, Network_Hyperparameters, Training_Hyperparameters, DataManager_Options)
            pretrained = torch.load(Training_Options['pretrained_network'])
            net.load_state_dict(pretrained['network'])

            # Alter final layer(s) of network to correctly output for new_classes
            Task_Options['classes'] = new_classes
            net = networks.edit_network(net, Task_Options, Network_Hyperparameters, Training_Hyperparameters, DataManager_Options)
    return net

def initialise_training_hyperparameters(Training_Hyperparameters, net, dataloader):
    """Initialises criterion, optimizer, and LR scheduler given user input parameters.

    For details on all input parameter dictionaries, see: 
    /spectrai/spectrai/configs/*.yml

    Arguments:
        Training_Hyperparameters: dictionary of training hyperparameters
        net: PyTorch neural network
        dataloader: PyTorch dataloader or None
    
    Returns:
        criterion: PyTorch criterion
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler or None
    """
    # ------------Criterion------------
    if Training_Hyperparameters['criterion'] == 'L1':
        criterion = nn.L1Loss()
    elif Training_Hyperparameters['criterion'] == 'L2 / MSE':
        criterion = nn.MSELoss()
    elif Training_Hyperparameters['criterion'] == 'Cross Entropy':
        criterion = nn.CrossEntropyLoss()
    else: # Binary Cross Entropy
        criterion = nn.BCELoss()

    # ------------Optimizer------------
    if Training_Hyperparameters['optimizer'] == "Adam":
        optimizer = optim.Adam(net.parameters(), lr = Training_Hyperparameters['learning_rate'])
    elif Training_Hyperparameters['optimizer'] == "Adagrad":
        optimizer = optim.Adagrad(net.parameters(), lr = Training_Hyperparameters['learning_rate'])
    elif Training_Hyperparameters['optimizer'] == "SGD":
        optimizer = optim.SGD(net.parameters(), lr = Training_Hyperparameters['learning_rate'])
    else: # RMSprop
        optimizer = optim.RMSprop(net.parameters(), lr = Training_Hyperparameters['learning_rate'])

    # ------------Scheduler------------
    if Training_Hyperparameters['scheduler'] == "Step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)
    elif Training_Hyperparameters['scheduler'] == "Multiplicative":
        lmbda = lambda epoch: 0.985
        scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
    elif Training_Hyperparameters['scheduler'] == "Cyclic":
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr = Training_Hyperparameters['learning_rate']/100, max_lr = Training_Hyperparameters['learning_rate'], mode = 'triangular2', cycle_momentum = False)
    elif Training_Hyperparameters['scheduler'] == "OneCycle" and dataloader is not None:
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = Training_Hyperparameters['learning_rate'], steps_per_epoch=len(dataloader), epochs=int(Training_Hyperparameters['epochs']), cycle_momentum = False)
    elif Training_Hyperparameters['scheduler'] == "ReduceOnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    else: # Constant
        scheduler = None

    return criterion, optimizer, scheduler

def initialise_dataset(Task_Options, Network_Hyperparameters, Training_Hyperparameters, Preprocessing, Data_Augmentation, DataManager_Options, apply = 0):
    """Initialises dataloaders given user input parameters.

    For details on all input parameter dictionaries, see: 
    /spectrai/spectrai/configs/*.yml

    Arguments:
        Task_Options: dictionary of task options
        Network_Hyperparameters: dictionary of network hyperparameters
        Training_Hyperparameters: dictionary of training hyperparameters
        Preprocessing: dictionary of preprocessing options
        Data_Augmentation: dictionary of data augmentation options
        DataManager_Options: dictionary of data manager options
        apply: a flag indicating that target data is unavailable for
            training/evaluation
    
    Returns:
        train_loader: PyTorch dataloader for train set or None
        val_loader: PyTorch dataloader for validation set or None
        test_loader: PyTorch dataloader for test set or None
    """
    Val_Data_Augmentation = {'horizontal_flip': 0, 'vertical_flip': 0, 'rotation': 0, 'random_crop': 0,
                            'spectral_shift': 0, 'spectral_flip': 0, 'spectral_background': 0, 'mixup': 0}
    Val_DataManager_Options = copy.deepcopy(DataManager_Options)
    Val_DataManager_Options['shuffle'] = 'False'

    if DataManager_Options['train_split'] != 'None' and (DataManager_Options['train_input_data'] != 'None' or DataManager_Options['train_target_data'] != 'None'):
        input_data, target_data, input_type, target_type, directory = dataloaders.prepare_data(DataManager_Options, Task_Options, 'train', apply)
        data, input_type, target_type, directory = dataloaders.split_data(Task_Options, DataManager_Options, input_data, target_data, input_type, target_type, directory)
        transform_list = dataloaders.prepare_transforms(Task_Options, Network_Hyperparameters, Training_Hyperparameters, Preprocessing, Data_Augmentation, DataManager_Options)
        val_transform_list = dataloaders.prepare_transforms(Task_Options, Network_Hyperparameters, Training_Hyperparameters, Preprocessing, Val_Data_Augmentation, Val_DataManager_Options)

        if data['train_input'] is not None or data['train_target'] is not None:
            train_loader = dataloaders.prepare_dataloader(Task_Options, Training_Hyperparameters, DataManager_Options, data['train_input'], data['train_target'], input_type, target_type, directory, transform_list, apply)
        else:
            train_loader = None
        if data['val_input'] is not None or data['val_target'] is not None:
            val_loader = dataloaders.prepare_dataloader(Task_Options, Training_Hyperparameters, DataManager_Options, data['val_input'], data['val_target'], input_type, target_type, directory, val_transform_list, apply)
        else:
            val_loader = None
        if data['test_input'] is not None or data['test_target'] is not None:
            test_loader = dataloaders.prepare_dataloader(Task_Options, Training_Hyperparameters, DataManager_Options, data['test_input'], data['test_target'], input_type, target_type, directory, val_transform_list, apply)
        else:
            test_loader = None

    else: #DataManager_Options['train_split'] == 'None'
        if DataManager_Options['train_input_data'] != 'None' or DataManager_Options['train_target_data'] != 'None':
            input_data, target_data, input_type, target_type, directory = dataloaders.prepare_data(DataManager_Options, Task_Options, 'train', apply)
            transform_list = dataloaders.prepare_transforms(Task_Options, Network_Hyperparameters, Training_Hyperparameters, Preprocessing, Data_Augmentation, DataManager_Options)
            train_loader = dataloaders.prepare_dataloader(Task_Options, Training_Hyperparameters, DataManager_Options, input_data, target_data, input_type, target_type, directory, transform_list, apply)
        else:
            train_loader = None  

        if DataManager_Options['val_input_data'] != 'None' or DataManager_Options['val_target_data'] != 'None':
            input_data, target_data, input_type, target_type, directory = dataloaders.prepare_data(DataManager_Options, Task_Options, 'val', apply)
            val_transform_list = dataloaders.prepare_transforms(Task_Options, Network_Hyperparameters, Training_Hyperparameters, Preprocessing, Val_Data_Augmentation, Val_DataManager_Options)
            val_loader = dataloaders.prepare_dataloader(Task_Options, Training_Hyperparameters, DataManager_Options, input_data, target_data, input_type, target_type, directory, val_transform_list, apply)
        else:
            val_loader = None

        if DataManager_Options['test_input_data'] != 'None' or DataManager_Options['test_target_data'] != 'None':
            input_data, target_data, input_type, target_type, directory = dataloaders.prepare_data(DataManager_Options, Task_Options, 'test', apply)
            val_transform_list = dataloaders.prepare_transforms(Task_Options, Network_Hyperparameters, Training_Hyperparameters, Preprocessing, Val_Data_Augmentation, Val_DataManager_Options)
            test_loader = dataloaders.prepare_dataloader(Task_Options, Training_Hyperparameters, DataManager_Options, input_data, target_data, input_type, target_type, directory, val_transform_list, apply)
        else:
            test_loader = None
    
    return train_loader, val_loader, test_loader