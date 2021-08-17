import numpy as np 
import scipy.io
import os
import sys
import pytest

import spectrai.trainer as trainer
import spectrai.previewer as previewer

def test_none():
    Data_Augmentation = {'horizontal_flip': 0,
                        'vertical_flip': 0,
                        'rotation': 0,
                        'random_crop': 0,
                        'spectral_shift': 0.0,
                        'spectral_flip': 0,
                        'spectral_background': 0,
                        'mixup': 0}

    Network_Hyperparameters = {'network': 'None',
                            'dimension': '2D',
                            'activation': 'None',
                            'normalization': 'None'}

    Training_Hyperparameters = {'epochs': 0,
                                'batch_size': 0,
                                'learning_rate': 0.0,
                                'input_image_size': 0,
                                'target_image_size': 0,
                                'spectrum_length': 0,
                                'optimizer': 'None',
                                'scheduler': 'None',
                                'criterion': 'None'}

    Preprocessing = {'spectral_crop_start': 0,
                        'spectral_crop_end': 500,
                        'background_subtraction': 'None',
                        'data_normalization': 'None'}

    Training_Options = {'training_option': 'None', 'pretrained_network': 'None'}

    Task_Options = {'task': 'None', 'classes': 0}

    DataManager_Options = {'data_format': 'None', 
                        'data_directory': 'None',
                        'train_input_data': 'None', 
                        'val_input_data': 'None', 
                        'test_input_data': 'None',
                        'train_target_data': 'None',
                        'val_target_data': 'None', 
                        'test_target_data': 'None',
                        'shuffle': 'None', 
                        'seed': 'None', 
                        'train_split': 'None', 
                        'val_split': 'None', 
                        'test_split': 'None'
                        }

    net_state_dict = 'None'
    optimizer_state_dict = 'None'
    scheduler_state_dict = 'None'

    with pytest.raises(ValueError):
        output = trainer.train_epoch(Training_Options, Task_Options, Network_Hyperparameters, 
                                    Training_Hyperparameters, Preprocessing, Data_Augmentation, 
                                    DataManager_Options, net_state_dict, optimizer_state_dict, 
                                    scheduler_state_dict, 1)