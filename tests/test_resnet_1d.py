import numpy as np 
import scipy.io
import os
import sys

import spectrai.trainer as trainer
import spectrai.previewer as previewer

DIR_ROOT = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DIR_DATA = os.path.join(DIR_ROOT, 'data/test_classification_spectra/')

def test_spectra_classification():
    Augmentation_List = {'horizontal_flip': 0,
                        'vertical_flip': 0,
                        'rotation': 0,
                        'random_crop': 0,
                        'spectral_shift': 0.0,
                        'spectral_flip': 1,
                        'spectral_background': 0,
                        'mixup': 1}

    Network_Hyperparameters = {'network': 'ResNet',
                            'dimension': '2D',
                            'activation': 'ReLU',
                            'normalization': 'None'}

    Training_Hyperparameters = {'epochs': 2,
                                'batch_size': 64,
                                'learning_rate': 0.0001,
                                'input_image_size': 16,
                                'target_image_size': 64,
                                'spectrum_length': 500,
                                'optimizer': 'Adam',
                                'scheduler': 'Constant',
                                'criterion': 'Cross Entropy'}

    Preprocessing_List = {'spectral_crop_start': 0,
                        'spectral_crop_end': 500,
                        'background_subtraction': 'None',
                        'data_normalization': 'Max Value'}

    Training_Options = {'training_option': 'Train From Scratch', 'pretrained_network': 'None'}

    Task_Options = {'task': 'Classification', 'classes': 2}

    DataManager_Options = {'data_format': 'Spectra', 
                        'data_directory': 'True',
                        'train_input_data': DIR_DATA, 
                        'val_input_data': 'None', 
                        'test_input_data': 'None',
                        'train_target_data': 'None',
                        'val_target_data': 'None', 
                        'test_target_data': 'None',
                        'shuffle': 'False', 
                        'seed': 'None', 
                        'train_split': 'None', 
                        'val_split': 'None', 
                        'test_split': 'None'
                        }

    net_state_dict = 'None'
    optimizer_state_dict = 'None'
    scheduler_state_dict = 'None'

    output = trainer.train_epoch(Training_Options, Task_Options, Network_Hyperparameters, 
                                Training_Hyperparameters, Preprocessing_List, Augmentation_List, 
                                DataManager_Options, net_state_dict, optimizer_state_dict, 
                                scheduler_state_dict, 1)