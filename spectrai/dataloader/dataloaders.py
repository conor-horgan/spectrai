import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import spectrai.utils.utilities as utilities
import spectrai.transforms.spectral_transforms as spectral_transforms
import spectrai.dataloader.spectral_dataset as spectral_dataset

def prepare_data(DataManager_Options, Task_Options, dataset_type, apply = 0):
    """Generates input and target data given user input parameters.

    For details on all input parameter dictionaries, see: 
    /spectrai/spectrai/configs/*.yml

    Arguments:
        DataManager_Options: dictionary of data manager options
        Task_Options: dictionary of task options
        dataset_type: variable indicating dataset as 'train', 'val'
            or 'test'
        apply: flag indicating that target data is unavailable for
            training/evaluation
    
    Returns:
        input_data: Pandas dataframe containing paths to input data
            samples and labels (where appropriate)
        target_data: Pandas dataframe containing paths to target data
            samples and labels (where appropriate)
        input_type: variable indicating input data as 'spectrum' or
            'hyperspectral_image'
        target_type: variable indicating target data as 'spectrum',
            'hyperspectral_image', 'mask', or 'class'
        directory: variable indicating whether data is a directory of
            files or a single input file
    """
    input_data, target_data = get_data(DataManager_Options, Task_Options, dataset_type, apply)

    if DataManager_Options['data_format'] == 'Spectra':
        input_type = 'spectrum'
    else:
        input_type = 'hyperspectral_image'

    if DataManager_Options['data_directory'] == 'True':
        directory = True
        input_data, target_data, input_type, target_type = load_data_from_directory(DataManager_Options, Task_Options, input_data, target_data, input_type, apply)
    else: #DataManager_Options['data_directory'] == 'False'
        directory = False
        input_data, target_data, input_type, target_type = load_data_from_input(DataManager_Options, Task_Options, input_data, target_data, input_type, apply)
    return input_data, target_data, input_type, target_type, directory

def get_data(DataManager_Options, Task_Options, dataset_type, apply = 0):
    """Defines input and target data given user input parameters.

    For details on all input parameter dictionaries, see: 
    /spectrai/spectrai/configs/*.yml

    Arguments:
        DataManager_Options: dictionary of data manager options
        Task_Options: dictionary of task options
        dataset_type: variable indicating dataset as 'train', 'val'
            or 'test'
        apply: flag indicating that target data is unavailable for
            training/evaluation
    
    Returns:
        input_data: path to input data
        target_data: path to target data
    """
    if dataset_type == 'train':
        input_data = DataManager_Options['train_input_data']
        target_data = DataManager_Options['train_target_data']
    elif dataset_type == 'val':
        input_data = DataManager_Options['val_input_data']
        target_data = DataManager_Options['val_target_data']
    elif dataset_type == 'test' and Task_Options['task'] == 'Super-Resolution':
        if apply:
            input_data = DataManager_Options['test_target_data']
            target_data = None
        else:
            input_data = None
            target_data = DataManager_Options['test_target_data']
    else: #dataset_type == 'test':
        input_data = DataManager_Options['test_input_data']
        if apply:
            target_data = None
        else:
            target_data = DataManager_Options['test_target_data']
    return input_data, target_data

def load_data_from_directory(DataManager_Options, Task_Options, input_data, target_data, input_type, apply = 0):
    """Collects input and target data paths given user input parameters.

    For details on all input parameter dictionaries, see: 
    /spectrai/spectrai/configs/*.yml

    Arguments:
        DataManager_Options: dictionary of data manager options
        Task_Options: dictionary of task options
        input_data: path to input data
        target_data: path to target data
        input_type: variable indicating data as 'spectrum' or
            'hyperspectral_image'
        apply: flag indicating that target data is unavailable for
            training/evaluation
    
    Returns:
        input_data: Pandas dataframe containing paths to input data
            samples and labels (where appropriate)
        target_data: Pandas dataframe containing paths to target data
            samples and labels (where appropriate)
        input_type: variable indicating input data as 'spectrum' or
            'hyperspectral_image'
        target_type: variable indicating target data as 'spectrum',
            'hyperspectral_image', 'mask', or 'class'
    """    
    if Task_Options['task'] == 'Classification':
        if apply:
            input_data = utilities.prepare_dataset_directory(input_data)
        else:
            input_data = utilities.prepare_classification_dataset(input_data, input_type)
            target_data = 'None'
        target_type = 'class'
    elif Task_Options['task'] == 'Super-Resolution':
        if apply:
            input_data = utilities.prepare_dataset_directory(input_data)
        else:
            input_data = 'None'
            target_data = utilities.prepare_dataset_directory(target_data)
        target_type = 'hyperspectral_image'
    elif Task_Options['task'] == 'Segmentation':
        input_data = utilities.prepare_dataset_directory(input_data)
        if not apply:
            target_data = utilities.prepare_dataset_directory(target_data)
        target_type = 'mask'
    else: #Task_Options['task'] == 'Calibration' or Task_Options['task'] == 'Denoising':
        input_data = utilities.prepare_dataset_directory(input_data)
        if not apply:
            target_data = utilities.prepare_dataset_directory(target_data)
        if DataManager_Options['data_format'] == 'Spectra':
            target_type = 'spectrum'
        else:
            target_type = 'hyperspectral_image'
    return input_data, target_data, input_type, target_type

def load_data_from_input(DataManager_Options, Task_Options, input_data, target_data, input_type, apply = 0):
    """Loads input and target data given user input parameters.

    For details on all input parameter dictionaries, see: 
    /spectrai/spectrai/configs/*.yml

    Arguments:
        DataManager_Options: dictionary of data manager options
        Task_Options: dictionary of task options
        input_data: path to input data
        target_data: path to target data
        input_type: variable indicating data as 'spectrum' or
            'hyperspectral_image'
        apply: flag indicating that target data is unavailable for
            training/evaluation
    
    Returns:
        input_data: Pandas dataframe containing input data
            samples and labels (where appropriate)
        target_data: Pandas dataframe containing target data
            samples and labels (where appropriate)
        input_type: variable indicating input data as 'spectrum' or
            'hyperspectral_image'
        target_type: variable indicating target data as 'spectrum',
            'hyperspectral_image', 'mask', or 'class'
    """    
    if Task_Options['task'] == 'Classification':
        target_type = 'class'
    elif Task_Options['task'] == 'Super-Resolution':
        target_type = 'hyperspectral_image'
    elif Task_Options['task'] == 'Segmentation':
        target_type = 'mask'
    else: #Task_Options['task'] == 'Calibration' or Task_Options['task'] == 'Denoising':
        if DataManager_Options['data_format'] == 'Spectra':
            target_type = 'spectrum'
        else:
            target_type = 'hyperspectral_image'

    if DataManager_Options['data_format'] == 'Spectra':
        input_data = utilities.prepare_dataset_from_input(input_data, 'spectrum')
        if not apply:
            target_data = utilities.prepare_dataset_from_input(target_data, target_type)
    else:
        input_data = utilities.prepare_dataset_from_input(input_data, 'hyperspectral_image')
        if not apply:
            target_data = utilities.prepare_dataset_from_input(target_data, target_type)
    return input_data, target_data, input_type, target_type

def split_data(Task_Options, DataManager_Options, input_data, target_data, input_type, target_type, directory):
    """Splits data into train/val/test sets given user input parameters.

    For details on all input parameter dictionaries, see: 
    /spectrai/spectrai/configs/*.yml

    Arguments:
        Task_Options: dictionary of task options
        DataManager_Options: dictionary of data manager options
        input_data: Pandas dataframe containing paths to input data
            samples and labels (where appropriate)
        target_data: Pandas dataframe containing paths to target data
            samples and labels (where appropriate)
        input_type: variable indicating input data as 'spectrum' or
            'hyperspectral_image'
        target_type: variable indicating target data as 'spectrum',
            'hyperspectral_image', 'mask', or 'class'
        directory: variable indicating whether data is a directory of
            files or a single input file
    
    Returns:
        data: dictionary containing
                train_input
                val_input
                test_input
                train_target
                val_target
                test_target
            where each dictionary value contains a Pandas dataframe 
            with paths to data samples and labels (where appropriate)
            or None
        input_type: variable indicating input data as 'spectrum' or
            'hyperspectral_image'
        target_type: variable indicating target data as 'spectrum',
            'hyperspectral_image', 'mask', or 'class'
        directory: variable indicating whether data is a directory of
            files or a single input file
    """
    data = dict()
    val_test_split = (DataManager_Options['val_split'] + DataManager_Options['test_split']) * 0.01
    test_split = DataManager_Options['test_split'] * 0.01

    if Task_Options['task'] == 'Super-Resolution' and isinstance(target_data, pd.DataFrame):
        y_train, y_val_test = train_test_split(target_data, test_size=val_test_split)
        data['train_input'] = None
        data['val_input'] = None
        data['test_input'] = None
        data['train_target'] = y_train.reset_index(drop=True)
    else:
        if isinstance(target_data, pd.DataFrame):
            x_train, x_val_test, y_train, y_val_test = train_test_split(input_data, target_data, test_size=val_test_split)
            data['train_target'] = y_train.reset_index(drop=True)
        else:
            x_train, x_val_test = train_test_split(input_data, test_size=val_test_split)
            data['train_target'] = None
        data['train_input'] = x_train.reset_index(drop=True)

    if test_split > 0.0:
        if Task_Options['task'] == 'Super-Resolution' and isinstance(target_data, pd.DataFrame):
            y_val, y_test = train_test_split(y_val_test, test_size=test_split/val_test_split)
            data['val_target'] = y_val.reset_index(drop=True)
            data['test_target'] = y_test.reset_index(drop=True)
        else:
            if isinstance(target_data, pd.DataFrame):
                x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=test_split/val_test_split)
                data['val_target'] = y_val.reset_index(drop=True)
                data['test_target'] = y_test.reset_index(drop=True)
            else:
                x_val, x_test = train_test_split(x_val_test, test_size=test_split/val_test_split)
                data['val_target'] = None
                data['test_target'] = None
            data['val_input'] = x_val.reset_index(drop=True)
            data['test_input'] = x_test.reset_index(drop=True)          
    else:
        if Task_Options['task'] == 'Super-Resolution' and isinstance(target_data, pd.DataFrame):
            data['val_target'] = y_val_test.reset_index(drop=True)
            data['test_target'] = None
        else:
            if isinstance(target_data, pd.DataFrame):
                data['val_target'] = y_val_test.reset_index(drop=True)
            else:
                data['val_target'] = None
            data['val_input'] = x_val_test.reset_index(drop=True)
            data['test_input'] = None
            data['test_target'] = None
    return data, input_type, target_type, directory

def prepare_transforms(Task_Options, Network_Hyperparameters, Training_Hyperparameters, Preprocessing, Data_Augmentation, DataManager_Options):
    """Composes PyTorch list of transforms given user input parameters.

    For details on all input parameter dictionaries, see: 
    /spectrai/spectrai/configs/*.yml

    Arguments:
        Task_Options: dictionary of task options
        Network_Hyperparameters: dictionary of network hyperparameters
        Training_Hyperparameters: dictionary of training hyperparameters
        Preprocessing: dictionary of preprocessing options
        Data_Augmentation: dictionary of data augmentation options
        DataManager_Options: dictionary of data manager options
        apply: flag indicating that target data is unavailable for
            training/evaluation
    
    Returns:
        transform_list: Composed PyTorch list of transforms (augmentations)
    """
    if DataManager_Options['data_format'] == 'Spectra':
        transform_list = prepare_spectral_transforms(Network_Hyperparameters,Training_Hyperparameters,
                                                    Preprocessing, Data_Augmentation)
    else:
        transform_list = prepare_image_transforms(Task_Options, Network_Hyperparameters, Training_Hyperparameters, Preprocessing, 
                                                    Data_Augmentation, DataManager_Options)
    return transforms.Compose(transform_list)

def prepare_spectral_transforms(Network_Hyperparameters, Training_Hyperparameters, Preprocessing, Data_Augmentation):
    spectrum_length = int(Training_Hyperparameters['spectrum_length'])
    spectral_crop_start = int(Preprocessing['spectral_crop_start'])
    spectral_crop_end = int(Preprocessing['spectral_crop_end'])
    background_subtraction = Preprocessing['background_subtraction']
    data_normalization = Preprocessing['data_normalization']
    spectral_shift = float(Data_Augmentation['spectral_shift'])
    spectral_flip = Data_Augmentation['spectral_flip']
    spectral_background = Data_Augmentation['spectral_background']

    transform_list = []

    transform_list.append(spectral_transforms.ToTensor())
    transform_list.append(spectral_transforms.Squeeze())
    transform_list.append(spectral_transforms.CropSpectrum(spectral_crop_start, spectral_crop_end))
    transform_list.append(spectral_transforms.PadSpectrum(spectrum_length))
    transform_list.append(spectral_transforms.ShiftSpectrum(spectral_shift))

    if spectral_flip:
        transform_list.append(spectral_transforms.FlipAxis(-1))

    if background_subtraction != 'None':
        if background_subtraction == '3rd Order Polynomial':
            transform_list.append(spectral_transforms.PolyBackgroundSpectrum(3))
        elif background_subtraction == '5th Order Polynomial':
            transform_list.append(spectral_transforms.PolyBackgroundSpectrum(5))
        elif background_subtraction == 'Minimum Value Offset':
            transform_list.append(spectral_transforms.MinBackgroundSpectrum())

    if data_normalization != 'None':
        if data_normalization == 'Max Value':
            transform_list.append(spectral_transforms.MaxNormalizeSpectrum())
        elif data_normalization == 'Area Under The Curve':
            transform_list.append(spectral_transforms.AUCNormalizeSpectrum())

    transform_list.append(spectral_transforms.AddDimLast())
    
    if Network_Hyperparameters['network'] != 'ResNet':
        transform_list.append(spectral_transforms.MakeChannelsLast())
    return transform_list

def prepare_image_transforms(Task_Options, Network_Hyperparameters, Training_Hyperparameters, Preprocessing, Data_Augmentation, DataManager_Options):
    target_image_size = int(Training_Hyperparameters['target_image_size'])
    input_image_size = int(Training_Hyperparameters['input_image_size'])
    spectrum_length = int(Training_Hyperparameters['spectrum_length'])
    spectral_crop_start = int(Preprocessing['spectral_crop_start'])
    spectral_crop_end = int(Preprocessing['spectral_crop_end'])
    background_subtraction = Preprocessing['background_subtraction']
    data_normalization = Preprocessing['data_normalization']
    spectral_shift = float(Data_Augmentation['spectral_shift'])
    spectral_flip = Data_Augmentation['spectral_flip']
    spectral_background = Data_Augmentation['spectral_background']
    horizontal_flip = Data_Augmentation['horizontal_flip']
    vertical_flip = Data_Augmentation['vertical_flip']
    rotation = Data_Augmentation['rotation']
    random_crop = Data_Augmentation['random_crop']

    transform_list = []

    transform_list.append(spectral_transforms.ToTensor())
    if DataManager_Options['data_format'] == 'Image: C, H, W':
        transform_list.append(spectral_transforms.MakeChannelsLast())

    transform_list.append(spectral_transforms.PadCropImage(target_image_size, random_crop))
    transform_list.append(spectral_transforms.CropImageSpectrum(spectral_crop_start, spectral_crop_end))
    transform_list.append(spectral_transforms.ShiftImageSpectrum(spectral_shift))

    if vertical_flip:
        transform_list.append(spectral_transforms.FlipAxis(0))
    if horizontal_flip:
        transform_list.append(spectral_transforms.FlipAxis(1))
    if spectral_flip:
        transform_list.append(spectral_transforms.FlipAxis(2))
    if rotation:
        transform_list.append(spectral_transforms.RotateImage())

    if background_subtraction != 'None':
        if background_subtraction == '3rd Order Polynomial':
            transform_list.append(spectral_transforms.PolyBackgroundImage(3))
        elif background_subtraction == '5th Order Polynomial':
            transform_list.append(spectral_transforms.PolyBackgroundImage(5))
        elif background_subtraction == 'Minimum Value Offset':
            transform_list.append(spectral_transforms.MinBackgroundImage())

    if data_normalization != 'None':
        if data_normalization == 'Max Value':
            transform_list.append(spectral_transforms.MaxNormalizeImage())
        elif data_normalization == 'Area Under The Curve':
            transform_list.append(spectral_transforms.AUCNormalizeImage())
    
    #if Task_Options['task'] == 'Super-Resolution' and not apply:
    if Task_Options['task'] == 'Super-Resolution':
        #scale = target_image_size // input_image_size
        #transform_list.append(spectral_transforms.SkipDownsampleImage(scale))
        transform_list.append(spectral_transforms.BicubicDownsampleImage(input_image_size))
    
    transform_list.append(spectral_transforms.MakeChannelsFirst())

    if Network_Hyperparameters['dimension'] == '3D':
        transform_list.append(spectral_transforms.AddDimFirst())
    return transform_list

def prepare_dataloader(Task_Options, Training_Hyperparameters, DataManager_Options, input_data, target_data, input_type, target_type, directory, transform_list, apply):
    """Creates PyTorch dataset and dataloader given user input parameters.

    For details on all input parameter dictionaries, see: 
    /spectrai/spectrai/configs/*.yml

    Arguments:
        Task_Options: dictionary of task options
        Network_Hyperparameters: dictionary of network hyperparameters
        DataManager_Options: dictionary of data manager options
        input_data: Pandas dataframe containing paths to input data
            samples and labels (where appropriate)
        target_data: Pandas dataframe containing paths to target data
            samples and labels (where appropriate)
        input_type: variable indicating input data as 'spectrum' or
            'hyperspectral_image'
        target_type: variable indicating target data as 'spectrum',
            'hyperspectral_image', 'mask', or 'class'
        directory: variable indicating whether data is a directory of
            files or a single input file
        transform_list: Composed PyTorch list of transforms (augmentations)
        apply: a flag indicating that target data is unavailable for
            training/evaluation
    
    Returns:
        user_loader: PyTorch dataloader
    """
    if Task_Options['task'] == 'Calibration':
        user_dataset = spectral_dataset.SpectralDataset(input_data, target_data, input_type, target_type, directory, transform_list, apply)
    elif Task_Options['task'] == 'Denoising':
        user_dataset = spectral_dataset.SpectralDataset(input_data, target_data, input_type, target_type, directory, transform_list, apply)
    elif Task_Options['task'] == 'Segmentation':
        user_dataset = spectral_dataset.ImageSegmentationDataset(input_data, target_data, input_type, target_type, directory, transform_list, apply)
    elif Task_Options['task'] == 'Super-Resolution':
        user_dataset = spectral_dataset.ImageSuperResDataset(input_data, target_data, input_type, target_type, directory, transform_list, apply)
    else: #Task_Options['task'] == 'Classification':
        user_dataset = spectral_dataset.ClassificationDataset(input_data, target_data, input_type, target_type, directory, transform_list, apply)
        
    if DataManager_Options['shuffle'] == 'True':
        user_loader = DataLoader(user_dataset, batch_size = int(Training_Hyperparameters['batch_size']), shuffle = True, num_workers = 0)
    else:
        user_loader = DataLoader(user_dataset, batch_size = int(Training_Hyperparameters['batch_size']), shuffle = False, num_workers = 0)
    return user_loader
