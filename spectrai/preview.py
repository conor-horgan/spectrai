import os
import torch
import sys
import argparse

module_path = os.path.abspath(os.path.join('../spectrai'))
if module_path not in sys.path:
    sys.path.append(module_path)

import spectrai.trainer as trainer
import spectrai.utils.utilities as utilities
import spectrai.initialise as initialise
import spectrai.previewer as previewer

parser = argparse.ArgumentParser(description='spectrai: preview sample preprocessing')

parser.add_argument('--config', default='image_superresolution.yml', type=str)

parser.add_argument('--task', type=str)
parser.add_argument('--classes', type=int)

parser.add_argument('--training_option', type=str)
parser.add_argument('--pretrained_network', type=str)
parser.add_argument('--pretrained_classes', type=int)

parser.add_argument('--network', type=str)
parser.add_argument('--activation', type=str)
parser.add_argument('--normalization', type=str)

parser.add_argument('--epochs', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--input_image_size', type=int)
parser.add_argument('--target_image_size', type=int)
parser.add_argument('--spectrum_length', type=int)
parser.add_argument('--optimizer', type=str)
parser.add_argument('--scheduler', type=str)
parser.add_argument('--criterion', type=str)

parser.add_argument('--spectral_crop_start', type=int)
parser.add_argument('--spectral_crop_end', type=int)
parser.add_argument('--background_subtraction', type=str)
parser.add_argument('--data_normalization', type=str)

parser.add_argument('--horizontal_flip', type=int)
parser.add_argument('--vertical_flip', type=int)
parser.add_argument('--rotation', type=int)
parser.add_argument('--random_crop', type=int)
parser.add_argument('--spectral_shift', type=float)
parser.add_argument('--spectral_flip', type=int)
parser.add_argument('--spectral_background', type=int)
parser.add_argument('--mixup', type=int)

parser.add_argument('--data_format', type=str)
parser.add_argument('--data_directory', type=str)
parser.add_argument('--train_input_data', type=str)
parser.add_argument('--val_input_data', type=str)
parser.add_argument('--test_input_data', type=str)
parser.add_argument('--train_target_data', type=str)
parser.add_argument('--val_target_data', type=str)
parser.add_argument('--test_target_data', type=str)
parser.add_argument('--shuffle', type=str)
parser.add_argument('--seed', type=int)
parser.add_argument('--train_split', type=float)
parser.add_argument('--val_split', type=float)
parser.add_argument('--test_split', type=float)

def main():
    args, unknown = parser.parse_known_args()
    if unknown is not None:
        for arg in unknown:
            print("Unknown argument detected: %s" %arg)
    config = utilities.get_config(os.path.join(module_path, 'spectrai/configs', args.config))

    if args is not None:
        for args_key in args.__dict__:
            if args.__dict__[args_key] is not None:
                for config_key in config.keys():
                    if args_key in config[config_key].keys():
                        config[config_key][args_key] = args.__dict__[args_key]

    Task_Options = config['Task_Options']
    Training_Options = config['Training_Options']
    Network_Hyperparameters = config['Network_Hyperparameters']
    Training_Hyperparameters = config['Training_Hyperparameters']
    Preprocessing = config['Preprocessing']
    Data_Augmentation = config['Data_Augmentation']
    DataManager_Options = config['DataManager_Options']
    net_state_dict = config['State_Dicts']['net_state_dict']

    for data_path in ['train_input_data', 'val_input_data', 'test_input_data', 'train_target_data', 'val_target_data', 'test_target_data']:
        if DataManager_Options[data_path] != 'None':
            DataManager_Options[data_path] = os.path.join(module_path, 'data', DataManager_Options[data_path])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = initialise.initialise_network(Training_Options, Task_Options, Network_Hyperparameters, 
                                        Training_Hyperparameters, DataManager_Options, net_state_dict)
    net.to(device)

    output = previewer.preview_preprocessing(Task_Options, Network_Hyperparameters, Training_Hyperparameters, 
                                        Preprocessing, Data_Augmentation, DataManager_Options)

if __name__ == '__main__':
    main()