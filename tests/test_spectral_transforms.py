import pytest
import numpy as np
import scipy.io
import torch
import os
import sys

DIR_ROOT = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DIR_DATA_1 = os.path.join(DIR_ROOT,'data/test_target_spectra/target_spectra.mat')
DIR_DATA_2 = os.path.join(DIR_ROOT,'data/test_images/cells/cell_1.mat')

import spectrai.transforms.spectral_transforms as st

def load_spectrum():
    output_data = scipy.io.loadmat(DIR_DATA_1)
    output_values = list(output_data.values())
    spec = output_values[3]
    return torch.from_numpy(spec[0,:])

def load_image():
    output_data = scipy.io.loadmat(DIR_DATA_2)
    output_values = list(output_data.values())
    spec = output_values[3]
    return torch.from_numpy(spec)

@pytest.fixture
def spectrum():
    spectrum = load_spectrum()
    return spectrum.unsqueeze(0)

@pytest.fixture
def image():
    return load_image()

@pytest.fixture
def image_sample():
    image = load_image()
    sample = {'input':image, 'target': {'data': image, 'type': 'hyperspectral_image'}, 'name':'Cell'}
    return sample

def test_squeeze(spectrum):
    assert st.Squeeze().apply_transform(spectrum).shape == (500,)

def test_crop_spectrum(spectrum):
    assert st.CropSpectrum(100,400).apply_transform(spectrum).shape == (1,300)

def test_pad_spectrum(spectrum):
    assert st.PadSpectrum(400).apply_transform(spectrum).shape == (1,400)
    assert st.PadSpectrum(500).apply_transform(spectrum).shape == (1,500)
    assert st.PadSpectrum(600).apply_transform(spectrum).shape == (1,600)

def test_shift_spectrum(spectrum):
    assert st.ShiftSpectrum(0.1).apply_transform(spectrum, int(np.round(-0.1*500))).shape == spectrum.shape
    assert st.ShiftSpectrum(0.1).apply_transform(spectrum, int(np.round(0.1*500))).shape == spectrum.shape

def test_minbackground_spectrum(spectrum):
    assert st.MinBackgroundSpectrum().apply_transform(spectrum).shape == spectrum.shape
    assert torch.amin(st.MinBackgroundSpectrum().apply_transform(spectrum)) == pytest.approx(0.0)

def test_polybackground_spectrum(spectrum):
    assert st.PolyBackgroundSpectrum(3).apply_transform(spectrum).shape == spectrum.shape

def test_alsbackground_spectrum(spectrum):
    assert st.ALSBackgroundSpectrum(False).apply_transform(spectrum).shape == spectrum.shape

def test_maxnormalize_spectrum(spectrum):
    assert st.MaxNormalizeSpectrum().apply_transform(spectrum).shape == spectrum.shape
    assert torch.amax(st.MaxNormalizeSpectrum().apply_transform(spectrum)) == 1

def test_aucnormalize_spectrum(spectrum):
    assert st.AUCNormalizeSpectrum().apply_transform(spectrum).shape == spectrum.shape
    assert torch.sum(st.AUCNormalizeSpectrum().apply_transform(spectrum)) == pytest.approx(1.0)

def test_flip_spectrum(spectrum):
    assert st.FlipAxis(-1).apply_transform(spectrum).shape == spectrum.shape

def test_add_dim_first(spectrum):
    assert st.AddDimFirst().apply_transform(spectrum).shape == (1,1,500)

def test_add_dim_last(spectrum):
    assert st.AddDimLast().apply_transform(spectrum).shape == (1,500,1)

def test_make_channels_first(image):
    assert st.MakeChannelsFirst().apply_transform(image).shape == (image.shape[2], image.shape[0], image.shape[1])

def test_crop_image_spectrum(image):
    assert st.CropImageSpectrum(100,400).apply_transform(image).shape == (image.shape[0], image.shape[1], 300)

def test_shift_image_spectrum(image):
    assert st.ShiftImageSpectrum(0.1).apply_transform(image, int(np.round(0.1*image.shape[2]))).shape == image.shape
    assert st.ShiftImageSpectrum(0.1).apply_transform(image, int(np.round(-0.1*image.shape[2]))).shape == image.shape

def test_pad_crop_image1(image_sample):
    output = st.PadCropImage(32, True)
    output_data = output(image_sample)
    assert output_data['target']['data'].shape == (32,32,500)

def test_pad_crop_image2(image_sample):
    output = st.PadCropImage(110, True)
    output_data = output(image_sample)
    assert output_data['target']['data'].shape == (110,110,500)

def test_pad_crop_image3(image_sample):
    output = st.PadCropImage(32, False)
    output_data = output(image_sample)
    assert output_data['target']['data'].shape == (32,32,500)

def test_pad_crop_image4(image_sample):
    output = st.PadCropImage(110, False)
    output_data = output(image_sample)
    assert output_data['target']['data'].shape == (110,110,500)

def test_flip_image(image):
    assert st.FlipAxis(0).apply_transform(image).shape == image.shape
    assert st.FlipAxis(1).apply_transform(image).shape == image.shape
    assert st.FlipAxis(2).apply_transform(image).shape == image.shape

def test_rotate_image(image):
    assert st.RotateImage().apply_transform(image, 0.2).shape == (image.shape[1], image.shape[0], image.shape[2])
    assert st.RotateImage().apply_transform(image, 0.4).shape == image.shape
    assert st.RotateImage().apply_transform(image, 0.6).shape == (image.shape[1], image.shape[0], image.shape[2])
    assert st.RotateImage().apply_transform(image, 0.8).shape == image.shape

def test_downsample_image(image):
    assert image.shape[0]//4 <= st.SkipDownsampleImage(4).apply_transform(image).shape[0] <= np.ceil(image.shape[0]/4)
    assert image.shape[1]//4 <= st.SkipDownsampleImage(4).apply_transform(image).shape[1] <= np.ceil(image.shape[1]/4)
    assert st.SkipDownsampleImage(3).apply_transform(image).shape == (image.shape[0]//3, image.shape[1]//3, image.shape[2])
    assert st.SkipDownsampleImage(2).apply_transform(image).shape == (image.shape[0]//2, image.shape[1]//2, image.shape[2])

def test_maxnormalize_image(image):
    assert st.MaxNormalizeImage().apply_transform(image).shape == image.shape
    assert torch.amax(st.MaxNormalizeImage().apply_transform(image)) == 1

def test_aucnormalize_image(image):
    assert st.AUCNormalizeImage().apply_transform(image).shape == image.shape
    assert torch.sum(st.AUCNormalizeImage().apply_transform(image)) == pytest.approx(image.shape[0]*image.shape[1])

def test_minbackground_image(image):
    assert st.MinBackgroundImage().apply_transform(image).shape == image.shape
    assert torch.amin(st.MinBackgroundImage().apply_transform(image)) == pytest.approx(0.0)

def test_polybackground_image(image):
    assert st.PolyBackgroundImage(3).apply_transform(image).shape == image.shape

def test_totensor(image):
    assert torch.is_tensor(st.ToTensor().apply_transform(image.numpy()))