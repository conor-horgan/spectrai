import os
import numpy as np
import scipy.io
from PIL import Image
from torch.utils.data import Dataset

class SpectralDataset(Dataset):
    """Base class defining a spectral dataset.

    Arguments:
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
        apply: flag indicating that target data is unavailable for
            training/evaluation
        transform: Composed PyTorch list of transforms (augmentations)
    """
    def __init__(self, input_data, target_data, input_type, target_type, directory, transform, apply):
        self.input_data = input_data
        self.target_data = target_data
        self.input_type = input_type
        self.target_type = target_type
        self.directory = directory
        self.transform = transform
        self.apply = apply
        self.on_epoch_end()

    def load_sample(self, sample_path):
        """Loads a .mat or .npy data sample.

        Arguments:
            path: path for sample to be loaded
        
        Returns:
            sample: .mat or .npy data sample
        """
        if isinstance(sample_path,str):
            _, extension = os.path.splitext(sample_path)       
            if extension == '.mat':       
                sample_data = scipy.io.loadmat(sample_path)
                sample_values = list(sample_data.values())
                sample = sample_values[3]
            elif extension == '.npy':
                sample = np.load(sample_path)
            else:
                raise ValueError('Input file with extension %s is not a valid file type' %extension)
            return sample
        else:
            return sample_path

    def get_name(self, sample_path, idx):
        """Returns the name of a file from an input path."""
        if isinstance(sample_path,str):
            return os.path.splitext(os.path.basename(sample_path))[0]
        else:
            return str(idx)

    def __getitem__(self, idx):
        """Returns a dictionary containing input and target data.

        Arguments:
            idx: index for sample to be loaded
        
        Returns:
            sample: dictionary containing the following fields
                input: (possibly augmented) input data
                target:
                    data: (possibly augmented) target data
                    type: target data type (e.g. 'spectrum', 'hyperspectral_image')
        """
        if self.directory:
            name = self.get_name(self.input_data['Data'][idx], idx)
            image = self.load_sample(self.input_data['Data'][idx])
            if not self.apply:
                target = self.load_sample(self.target_data['Data'][idx])
        else:
            name = str(idx)
            image = self.input_data['Data'][idx]
            if not self.apply:
                target = self.target_data['Data'][idx]

        if self.apply:
            sample = {'input': image, 'target': {'data':image, 'type':self.input_type}, 'name': name}
        else:
            sample = {'input': image, 'target': {'data':target, 'type':self.target_type}, 'name': name}

        if self.transform:
            sample = self.transform(sample)
        
        return sample

    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return len(self.input_data)

class ClassificationDataset(SpectralDataset):
    """Class defining a spectral classification dataset."""
    def __init__(self, input_data, target_data, input_type, target_type, directory, transform, apply):
        super(ClassificationDataset, self).__init__(input_data, target_data, input_type, target_type, directory, transform, apply)
        
    def __getitem__(self, idx):
        name = self.get_name(self.input_data['Data'][idx], idx)
        spectrum = self.load_sample(self.input_data['Data'][idx])
        if not self.apply:
            target = self.input_data['Encoded_Labels'][idx]
        
        if self.apply:
            sample = {'input': spectrum, 'target': {'data':spectrum, 'type':self.input_type}, 'name': name}
        else:
            sample = {'input': spectrum, 'target': {'data':target, 'type':self.target_type}, 'name': name}

        if self.transform:
            sample = self.transform(sample)
                
        return sample
    
    def on_epoch_end(self):
        return super(ClassificationDataset, self).on_epoch_end()
    
    def __len__(self):
        return super(ClassificationDataset, self).__len__()

class ImageSuperResDataset(SpectralDataset):
    """Class defining a spectral image super-resolution dataset."""
    def __init__(self, input_data, target_data, input_type, target_type, directory, transform, apply):
        super(ImageSuperResDataset, self).__init__(input_data, target_data, input_type, target_type, directory, transform, apply)

    def __getitem__(self, idx):
        if self.apply:
            if self.directory:
                name = self.get_name(self.input_data['Data'][idx], idx)
                image = self.load_sample(self.input_data['Data'][idx])
            else:
                name = str(idx)
                image = self.input_data['Data'][idx]
        else:
            if self.directory:
                name = self.get_name(self.target_data['Data'][idx], idx)
                image = self.load_sample(self.target_data['Data'][idx])
            else:
                name = str(idx)
                image = self.target_data['Data'][idx]

            target = image
        
        if self.apply:
            sample = {'input': image, 'target': {'data':image, 'type':self.input_type}, 'name': name}
        else:
            sample = {'input': image, 'target': {'data':target, 'type':self.target_type}, 'name': name}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def on_epoch_end(self):
        return super(ImageSuperResDataset, self).on_epoch_end()
    
    def __len__(self):
        if self.apply:
            return len(self.input_data)
        else:
            return len(self.target_data)

class ImageSegmentationDataset(SpectralDataset):
    """Class defining a spectral image segmentation dataset."""
    def __init__(self, input_data, target_data, input_type, target_type, directory, transform, apply):
        super(ImageSegmentationDataset, self).__init__(input_data, target_data, input_type, target_type, directory, transform, apply)
        
    def load_mask(self, path):
        """Loads a mask image file.

        Arguments:
            path: path for sample to be loaded
        
        Returns:
            mask: mask numpy array
        """
        mask_image = Image.open(path)
        mask = np.asarray(mask_image)
        mask.astype(np.float64)
        mask = mask - 1
        return mask

    def __getitem__(self, idx):
        if self.directory:
            name = self.get_name(self.input_data['Data'][idx], idx)
            image = self.load_sample(self.input_data['Data'][idx])
            if not self.apply:
                target = self.load_mask(self.target_data['Data'][idx])
        else:
            name = str(idx)
            image = self.input_data['Data'][idx]
            if not self.apply:
                target = self.target_data['Data'][idx]
        
        if self.apply:
            sample = {'input': image, 'target': {'data':image, 'type':self.input_type}, 'name': name}
        else:
            sample = {'input': image, 'target': {'data':target, 'type':self.target_type}, 'name': name}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def on_epoch_end(self):
        return super(ImageSegmentationDataset, self).on_epoch_end()
    
    def __len__(self):
        return super(ImageSegmentationDataset, self).__len__()