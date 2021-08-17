import numpy as np
from scipy import sparse, ndimage
from scipy.sparse.linalg import spsolve
import torch
from torch import nn

class Transform(object):
    """Base class defining a spectral transform."""
    def __init__(self):
        pass

    def apply_transform(self, sample):
        return NotImplementedError

    def __call__(self, sample):
        """Applies transformation to input and (possibly) target data.

        Arguments:
            sample: dictionary containing the following fields
                input: (possibly augmented) input data
                target:
                    data: (possibly augmented) target data
                    type: target data type (e.g. 'spectrum', 'hyperspectral_image')
        
        Returns:
            sample: dictionary containing the following fields
                input: (possibly augmented) input data
                target:
                    data: (possibly augmented) target data
                    type: target data type (e.g. 'spectrum', 'hyperspectral_image')
        """
        sample_input, target_data, target_type, name = sample['input'], sample['target']['data'], sample['target']['type'], sample['name']

        sample_input = self.apply_transform(sample_input)

        if target_type == 'spectrum' or target_type == 'hyperspectral_image':
            target_data = self.apply_transform(target_data)

        return {'input': sample_input, 'target': {'data': target_data, 'type': target_type}, 'name': name}

class Squeeze(Transform):
    def __init__(self):
        super(Squeeze, self).__init__()

    def apply_transform(self, spectrum):
        spectrum = torch.squeeze(spectrum)
        return spectrum

    def __call__(self, sample):
        return super(Squeeze, self).__call__(sample)

class CropSpectrum(Transform):
    """Transform class that crops spectrum to given index range."""
    def __init__(self, start, end):
        super(CropSpectrum, self).__init__()
        assert isinstance(start, int)
        assert isinstance(end, int)
        self.start = start
        self.end = end

    def apply_transform(self, spectrum):
        spectrum = spectrum[...,self.start:self.end]
        return spectrum

    def __call__(self, sample):
        return super(CropSpectrum, self).__call__(sample)

class PadSpectrum(Transform):
    """Transform class that pads or crops spectrum to given length."""
    def __init__(self, spectrum_length):
        super(PadSpectrum, self).__init__()
        assert isinstance(spectrum_length, int)
        self.spectrum_length = spectrum_length

    def apply_transform(self, spectrum):
        if spectrum.shape[-1] == self.spectrum_length:
            spectrum = spectrum
        elif spectrum.shape[-1] > self.spectrum_length:
            spectrum = spectrum[...,0:self.spectrum_length]
        else:
            m = nn.ReflectionPad1d((0,self.spectrum_length - spectrum.shape[-1]))
            spectrum = m(spectrum.unsqueeze(0)).squeeze(0)
        return spectrum

    def __call__(self, sample):
        return super(PadSpectrum, self).__call__(sample)

class ShiftSpectrum(Transform):
    """Transform class that shifts spectrum left or right."""
    def __init__(self, shift):
        super(ShiftSpectrum, self).__init__()        
        assert isinstance(shift, float)
        self.shift = shift

    def apply_transform(self, spectrum, shift_amount):
        if shift_amount > 0:
            m = nn.ReflectionPad1d((0,abs(shift_amount)))
            if len(spectrum.shape) == 1:
                spectrum = m(spectrum[...,shift_amount:].unsqueeze(0).unsqueeze(0)).squeeze(0)
            else:
                spectrum = m(spectrum[...,shift_amount:].unsqueeze(0))
        elif shift_amount < 0:
            m = nn.ReflectionPad1d((abs(shift_amount),0))
            if len(spectrum.shape) == 1:
                spectrum = m(spectrum[...,:shift_amount].unsqueeze(0).unsqueeze(0)).squeeze(0)
            else:
                spectrum = m(spectrum[...,:shift_amount].unsqueeze(0))
        return spectrum.squeeze(0)

    def __call__(self, sample):
        sample_input, target_data, target_type, name = sample['input'], sample['target']['data'], sample['target']['type'], sample['name']

        if self.shift != 0.0:
            shift_range = np.random.uniform(-self.shift, self.shift)
            shift_amount = int(np.round(shift_range*sample_input.shape[-1]))

            sample_input = self.apply_transform(sample_input, shift_amount)

        if target_type == 'spectrum' or target_type == 'hyperspectral_image':
            if self.shift != 0.0:
                target_data = self.apply_transform(target_data, shift_amount)

        return {'input': sample_input, 'target': {'data': target_data, 'type': target_type}, 'name': name}

class MinBackgroundSpectrum(Transform):
    """Transform class that performs spectral minimum value background subtraction."""
    def __init__(self):
        super(MinBackgroundSpectrum, self).__init__()

    def apply_transform(self, spectrum):
        spectrum = spectrum - torch.amin(spectrum)
        return spectrum

    def __call__(self, sample):
        return super(MinBackgroundSpectrum, self).__call__(sample)

class PolyBackgroundSpectrum(Transform):
    """Transform class that performs spectral polynomial background subtraction."""
    def __init__(self, order):
        super(PolyBackgroundSpectrum, self).__init__()
        assert isinstance(order, int)
        self.order = order

    def apply_transform(self, spectrum):
        spectrum = np.squeeze(spectrum.numpy())
        x = np.arange(0, spectrum.shape[0])
        poly = np.poly1d(np.polyfit(x, spectrum, self.order))
        spectrum = spectrum - poly(x)
        if np.amin(spectrum) < 0.0:
            spectrum = spectrum + np.abs(np.amin(spectrum))
        spectrum = spectrum[np.newaxis,...]
        return torch.from_numpy(spectrum)

    def __call__(self, sample):
        return super(PolyBackgroundSpectrum, self).__call__(sample)

class ALSBackgroundSpectrum(Transform):
    """Transform class that performs asymmetric least squares background subtraction."""
    def __init__(self, apply_target, lam = 10000, p = 0.05, n = 10):
        assert isinstance(apply_target, bool)
        assert isinstance(lam, int)
        assert isinstance(p, float)
        assert isinstance(n, int)
        self.apply_target = apply_target
        self.lam = lam
        self.p = p
        self.n = n

    def apply_transform(self, spectrum):
        spectrum = np.squeeze(spectrum.numpy())
        L = len(spectrum)
        D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
        w = np.ones(L)
        for i in range(self.n):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + self.lam * D.dot(D.transpose())
            z = spsolve(Z, w*spectrum)
            w = self.p * (spectrum > z) + (1-self.p) * (spectrum < z)
        spectrum = spectrum - z
        spectrum = spectrum[np.newaxis,...]
        return torch.from_numpy(spectrum)

    def __call__(self, sample):
        return super(ALSBackgroundSpectrum, self).__call__(sample)

class MaxNormalizeSpectrum(Transform):
    """Transform class that normalizes a spectrum by its maximum value."""
    def __init__(self):
        super(MaxNormalizeSpectrum, self).__init__()

    def apply_transform(self, spectrum):
        spectrum = spectrum/torch.amax(spectrum)
        return spectrum

    def __call__(self, sample):
        return super(MaxNormalizeSpectrum, self).__call__(sample)

class AUCNormalizeSpectrum(Transform):
    """Transform class that normalizes a spectrum by the area under the curve."""
    def __init__(self):
        super(AUCNormalizeSpectrum, self).__init__()

    def apply_transform(self, spectrum):
        spectrum = spectrum/torch.sum(spectrum)
        return spectrum

    def __call__(self, sample):
        return super(AUCNormalizeSpectrum, self).__call__(sample)

class FlipAxis(Transform):
    """Transform class that flips one axis of an input sample."""
    def __init__(self, axis):
        super(FlipAxis, self).__init__()
        assert isinstance(axis, int)
        self.axis = axis

    def apply_transform(self, sample):
        sample = torch.flip(sample, [self.axis])
        return sample

    def __call__(self, sample):
        sample_input, target_data, target_type, name = sample['input'], sample['target']['data'], sample['target']['type'], sample['name']
        
        if torch.rand(1) < 0.5:
            sample_input = self.apply_transform(sample_input)

            if target_type == 'spectrum' or target_type == 'hyperspectral_image' or (target_type == 'mask' and self.axis != 2):
                target_data = self.apply_transform(target_data)

        return {'input': sample_input, 'target': {'data': target_data, 'type': target_type}, 'name': name}

class CropImageSpectrum(Transform):
    """Transform class that crops image spectra to given index range."""
    def __init__(self, start, end):
        super(CropImageSpectrum, self).__init__()
        assert isinstance(start, int)
        assert isinstance(end, int)
        self.start = start
        self.end = end

    def apply_transform(self, image):
        image = image[:,:,self.start:self.end]
        return image

    def __call__(self, sample):
        return super(CropImageSpectrum, self).__call__(sample)

class ShiftImageSpectrum(Transform):
    """Transform class that shifts image spectra left or right."""
    def __init__(self, shift):
        super(ShiftImageSpectrum, self).__init__()
        assert isinstance(shift, float)
        self.shift = shift

    def apply_transform(self, image, shift_amount):
        shifted_spectrum_image = image.unsqueeze(0)
        if shift_amount > 0:
            m = nn.ReflectionPad2d((0,abs(shift_amount),0,0))
            shifted_spectrum_image = m(shifted_spectrum_image[:,:,:,shift_amount:])
        elif shift_amount < 0:
            m = nn.ReflectionPad2d((abs(shift_amount),0,0,0))
            shifted_spectrum_image = m(shifted_spectrum_image[:,:,:,:shift_amount])
        return shifted_spectrum_image.squeeze()

    def __call__(self, sample):
        sample_input, target_data, target_type, name = sample['input'], sample['target']['data'], sample['target']['type'], sample['name']
        
        if self.shift != 0.0:
            shift_range = np.random.uniform(-self.shift, self.shift)
            shift_amount = int(np.round(shift_range*sample_input.shape[-1]))

            sample_input = self.apply_transform(sample_input, shift_amount)

        if target_type == 'spectrum' or target_type == 'hyperspectral_image':
            if self.shift != 0.0:
                target_data = self.apply_transform(target_data, shift_amount)

        return {'input': sample_input, 'target': {'data': target_data, 'type': target_type}, 'name': name}

class PadCropImage(Transform):
    """Transform class that pads or crops image spectra to given length."""
    def __init__(self, size, random_crop):
        super(PadCropImage, self).__init__()
        assert isinstance(size, int)
        self.size = size
        self.random_crop = random_crop

    def random_crop_image(self, image, image_size, start_idx_x, start_idx_y, target_type):
        """Returns a random cropped patch of input image"""                   
        if image.shape[0] > image_size:
            end_idx_x = start_idx_x + image_size
        else:
            start_idx_x = 0
            end_idx_x = image.shape[0]

        if image.shape[1] > image_size:
            end_idx_y = start_idx_y + image_size
        else:
            start_idx_y = 0
            end_idx_y = image.shape[1]

        if target_type == 'mask':
            image_patch = image[start_idx_x:end_idx_x,start_idx_y:end_idx_y]
        else:
            image_patch = image[start_idx_x:end_idx_x,start_idx_y:end_idx_y,:]
        return image_patch

    def center_crop_image(self, image, image_size, target_type):
        """Returns central cropped patch of input image"""    
        cropped_image = image
        if image.shape[0] > image_size:
            dif = int(np.floor((image.shape[0] - image_size)/2))
            if target_type == 'mask':
                cropped_image = cropped_image[dif:image_size+dif,:]
            else:
                cropped_image = cropped_image[dif:image_size+dif,:,:]

        if image.shape[1] > image_size:
            dif = int(np.floor((image.shape[1] - image_size)/2))
            if target_type == 'mask':
                cropped_image = cropped_image[:,dif:image_size+dif]
            else:
                cropped_image = cropped_image[:,dif:image_size+dif,:]
        return cropped_image

    def apply_transform(self, image, start_idx_x, start_idx_y, target_type):
        if image.shape[0] == self.size and image.shape[1] == self.size:
            padded_image = image
        elif image.shape[0] > self.size and image.shape[1] > self.size:
            if self.random_crop:
                padded_image = self.random_crop_image(image, self.size, start_idx_x, start_idx_y, target_type)
            else:
                padded_image = self.center_crop_image(image, self.size, target_type)                    
        else:
            padded_image = image
            if padded_image.shape[0] > self.size:
                if self.random_crop:
                    padded_image = self.random_crop_image(padded_image, self.size, start_idx_x, start_idx_y, target_type)
                else:
                    padded_image = self.center_crop_image(padded_image, self.size, target_type) 
            else:           
                pad_before = int(np.floor((self.size - padded_image.shape[0])/2))
                pad_after = int(np.ceil((self.size - padded_image.shape[0])/2))
                if target_type == 'hyperspectral_image':
                    m = nn.ReflectionPad2d((0,0,pad_before,pad_after))
                    padded_image = m(torch.movedim(padded_image,-1,0).unsqueeze(0))
                    padded_image = torch.movedim(padded_image.squeeze(),0,-1)
                elif target_type == 'mask':
                    m = nn.ReflectionPad2d((pad_before,pad_after,0,0))
                    padded_image = m(torch.movedim(padded_image,-1,0).unsqueeze(0).unsqueeze(0))
                    padded_image = torch.movedim(padded_image.squeeze(),0,-1)

            if padded_image.shape[1] > self.size:
                if self.random_crop:
                    padded_image = self.random_crop_image(padded_image, self.size, start_idx_x, start_idx_y, target_type)
                else:
                    padded_image = self.center_crop_image(padded_image, self.size, target_type) 
            else:           
                pad_before = int(np.floor((self.size - padded_image.shape[1])/2))
                pad_after = int(np.ceil((self.size - padded_image.shape[1])/2))
                if target_type == 'hyperspectral_image':
                    m = nn.ReflectionPad2d((pad_before,pad_after,0,0))
                    padded_image = m(torch.movedim(padded_image,-1,0).unsqueeze(0))
                    padded_image = torch.movedim(padded_image.squeeze(),0,-1)
                elif target_type == 'mask':
                    m = nn.ReflectionPad2d((0,0,pad_before,pad_after))
                    padded_image = m(torch.movedim(padded_image,-1,0).unsqueeze(0).unsqueeze(0))
                    padded_image = torch.movedim(padded_image.squeeze(),0,-1)

        return padded_image

    def __call__(self, sample):
        sample_input, target_data, target_type, name = sample['input'], sample['target']['data'], sample['target']['type'], sample['name']
        
        start_idx_x = int(np.round(np.random.random() * (sample_input.shape[0]-self.size)))
        start_idx_y = int(np.round(np.random.random() * (sample_input.shape[1]-self.size)))
        
        sample_input = self.apply_transform(sample_input, start_idx_x, start_idx_y, 'hyperspectral_image')

        if target_type == 'hyperspectral_image' or target_type == 'mask':
            target_data = self.apply_transform(target_data, start_idx_x, start_idx_y, target_type)

        return {'input': sample_input, 'target': {'data': target_data, 'type': target_type}, 'name': name}

class RotateImage(Transform):
    """Transform class that performs x 90 degree rotations of an image."""
    def __init__(self):
        super(RotateImage, self).__init__()

    def apply_transform(self, image, rotation_extent):
        if rotation_extent < 0.25:
            rotation = 1
        elif rotation_extent < 0.5:
            rotation = 2
        elif rotation_extent < 0.75:
            rotation = 3
        else:
            rotation = 0
        image = torch.rot90(image, rotation)
        return image

    def __call__(self, sample):
        sample_input, target_data, target_type, name = sample['input'], sample['target']['data'], sample['target']['type'], sample['name']
        
        rotation_extent = torch.rand(1)

        sample_input = self.apply_transform(sample_input, rotation_extent)

        if target_type == 'hyperspectral_image' or target_type == 'mask':
            target_data = self.apply_transform(target_data, rotation_extent)

        return {'input': sample_input, 'target': {'data': target_data, 'type': target_type}, 'name': name}

class SkipDownsampleImage(Transform):
    """Transform class that performs skip spatial downsampling of an image."""
    def __init__(self, scale):
        super(SkipDownsampleImage, self).__init__()
        assert isinstance(scale, int)
        self.scale = scale

    def apply_transform(self, image):
        if self.scale >= 4:
            start_idx = torch.randint(1,self.scale-1, (1,))
        else:
            start_idx = 1 
        image = image[start_idx::self.scale,start_idx::self.scale,:]
        return image

    def __call__(self, sample):
        sample_input, target_data, target_type, name = sample['input'], sample['target']['data'], sample['target']['type'], sample['name']

        sample_input = self.apply_transform(sample_input)

        return {'input': sample_input, 'target': {'data': target_data, 'type': target_type}, 'name': name}

class BicubicDownsampleImage(Transform):
    """Transform class that performs bicubic spatial downsampling of an image."""
    def __init__(self, size):
        super(BicubicDownsampleImage, self).__init__()
        assert isinstance(size, int)
        self.size = size

    def apply_transform(self, image):
        image = torch.movedim(image, -1, 0).unsqueeze(0)
        image = nn.functional.interpolate(image, size = (self.size,self.size), mode = 'bicubic', align_corners = False)
        image = torch.movedim(image, 1, -1).squeeze(0)
        return image

    def __call__(self, sample):
        sample_input, target_data, target_type, name = sample['input'], sample['target']['data'], sample['target']['type'], sample['name']

        sample_input = self.apply_transform(sample_input)

        return {'input': sample_input, 'target': {'data': target_data, 'type': target_type}, 'name': name}

class MaxNormalizeImage(Transform):
    """Transform class that normalizes image spectra by the maximum image spectral value."""
    def __init__(self):
        super(MaxNormalizeImage, self).__init__()

    def apply_transform(self, image):
        image_max = torch.amax(image).repeat(image.shape)
        image = torch.div(image, image_max)
        return image

    def __call__(self, sample):
        return super(MaxNormalizeImage, self).__call__(sample)

class AUCNormalizeImage(Transform):
    """Transform class that normalizes image spectra by the area under the curve."""
    def __init__(self):
        super(AUCNormalizeImage, self).__init__()

    def apply_transform(self, image):
        image_sum = torch.sum(image,-1).unsqueeze(-1)
        sum_tile = torch.repeat_interleave(image_sum, image.shape[-1], -1)
        image = torch.div(image, sum_tile)
        return image

    def __call__(self, sample):
        return super(AUCNormalizeImage, self).__call__(sample)

class MinBackgroundImage(Transform):
    """Transform class that performs spectral image minimum value background subtraction."""
    def __init__(self):
        super(MinBackgroundImage, self).__init__()

    def apply_transform(self, image):
        min_values = torch.amin(image, 2)
        min_values = torch.clamp(min_values,min=0.0,max=torch.amax(min_values)).unsqueeze(-1)
        min_tile = torch.repeat_interleave(min_values, image.shape[-1], -1)
        image = image - min_tile
        return image

    def __call__(self, sample):
        return super(MinBackgroundImage, self).__call__(sample)

class PolyBackgroundImage(Transform):
    """Transform class that performs spectral image polynomial background subtraction."""
    def __init__(self, order):
        super(PolyBackgroundImage, self).__init__()
        assert isinstance(order, int)
        self.order = order

    def apply_transform(self, image):
        image = image.numpy()
        reshaped_image = np.reshape(image, ((image.shape[0]*image.shape[1], image.shape[2]))).T
        x = np.arange(0, image.shape[2])
        poly = np.polyfit(x, reshaped_image, self.order)
        image_out = reshaped_image.T
        for i in range(image_out.shape[0]):
            poly_1d = np.poly1d(poly[:,i])
            image_out[i] = image_out[i] - poly_1d(x)
            if np.amin(image_out[i,:]) < 0.0:
                image_out[i,:] = image_out[i,:] + np.abs(np.amin(image_out[i,:]))
        image_out = np.reshape(image_out, ((image.shape[0], image.shape[1], image.shape[2])))
        return torch.from_numpy(image_out)

    def __call__(self, sample):
        return super(PolyBackgroundImage, self).__call__(sample)

class MakeChannelsFirst(Transform):
    """Transform class that converts an image from channels last to channels first."""
    def __init__(self):
        super(MakeChannelsFirst, self).__init__()

    def apply_transform(self, sample):
        sample = torch.movedim(sample, -1, 0)
        return sample

    def __call__(self, sample):
        return super(MakeChannelsFirst, self).__call__(sample)

class MakeChannelsLast(Transform):
    """Transform class that converts an image from channels first to channels last."""
    def __init__(self):
        super(MakeChannelsLast, self).__init__()

    def apply_transform(self, sample):
        sample = torch.movedim(sample, 0, -1)
        return sample

    def __call__(self, sample):
        return super(MakeChannelsLast, self).__call__(sample)

class AddDimFirst(Transform):
    """Transform class that adds a dimension at the start."""
    def __init__(self):
        super(AddDimFirst, self).__init__()

    def apply_transform(self, sample):
        sample = sample.unsqueeze(0)
        return sample

    def __call__(self, sample):
        return super(AddDimFirst, self).__call__(sample)

class AddDimLast(Transform):
    """Transform class that adds a dimension at the end."""
    def __init__(self):
        super(AddDimLast, self).__init__()

    def apply_transform(self, sample):
        sample = sample.unsqueeze(-1)
        return sample

    def __call__(self, sample):
        return super(AddDimLast, self).__call__(sample)

class ToTensor(Transform):
    """Transform class that converts a numpy array to a torch tensor."""
    def __init__(self):
        super(ToTensor, self).__init__()

    def apply_transform(self, image):
        return torch.from_numpy(image).double()
    
    def __call__(self, sample):
        sample_input, target_data, target_type, name = sample['input'], sample['target']['data'], sample['target']['type'], sample['name']

        sample_input = self.apply_transform(sample_input)

        if target_type == 'spectrum' or target_type == 'hyperspectral_image' or target_type == 'mask':
            target_data = self.apply_transform(target_data)

        return {'input': sample_input, 'target': {'data': target_data, 'type': target_type}, 'name': name}

class ToNumpy(Transform):
    """Transform class that converts a torch tensor to a numpy array."""
    def __init__(self):
        super(ToNumpy, self).__init__()

    def apply_transform(self, image):
        return image.numpy()
    
    def __call__(self, sample):
        sample_input, target_data, target_type, name = sample['input'], sample['target']['data'], sample['target']['type'], sample['name']

        sample_input = self.apply_transform(sample_input)

        if target_type == 'spectrum' or target_type == 'hyperspectral_image' or target_type == 'mask':
            target_data = self.apply_transform(target_data)

        return {'input': sample_input, 'target': {'data': target_data, 'type': target_type}, 'name': name}