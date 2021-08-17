import os
import spectrai.trainer as trainer

DIR_ROOT = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DIR_DATA = os.path.join(DIR_ROOT, 'data/test_images/cells/')

def test_image_superres():
    Augmentation_List = {'horizontal_flip': 1,
                        'vertical_flip': 1,
                        'rotation': 1,
                        'random_crop': 1,
                        'spectral_shift': 0.1,
                        'spectral_flip': 1,
                        'spectral_background': 0,
                        'mixup': 1}

    Network_Hyperparameters = {'network': 'ResUNet',
                            'dimension': '2D',
                            'activation': 'PReLU',
                            'normalization': 'None'}

    Training_Hyperparameters = {'epochs': 1,
                                'batch_size': 2,
                                'learning_rate': 0.0001,
                                'input_image_size': 16,
                                'target_image_size': 64,
                                'spectrum_length': 250,
                                'optimizer': 'Adam',
                                'scheduler': 'Constant',
                                'criterion': 'L1'}

    Preprocessing_List = {'spectral_crop_start': 100,
                        'spectral_crop_end': 350,
                        'background_subtraction': 'None',
                        'data_normalization': 'Max Value'}

    Training_Options = {'training_option': 'Train From Scratch', 'pretrained_network': 'None'}

    Task_Options = {'task': 'Super-Resolution', 'classes': 2}

    DataManager_Options = {'data_format': 'Image: H, W, C', 
                        'data_directory': 'True',
                        'train_input_data': 'None', 
                        'val_input_data': 'None', 
                        'test_input_data': 'None',
                        'train_target_data': DIR_DATA,
                        'val_target_data': 'None', 
                        'test_target_data': 'None',
                        'shuffle': 'True', 
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
