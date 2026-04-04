import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
import io
import os
import glob
import torch

def image_processor(image_file):
    """in back end when an image gets uploaded we might primarely only use this function
       """
    # STEP 1: first we use PILL as the front door to accept different file formats and turn them all into RGB
    try:
        pil_image = Image.open(image_file).convert('RGB')
    except Exception as e:
        print("Failed to load image: {e}")
        return None, None
    # STEP 2: we turn the image into randomly cropped 256*256 size images preserving original resolution and pass it on as a tensor
    patch_pipline = transforms.Compose([
        transforms.RandomCrop((256,256)),
        transforms.ToTensor(),
    ])

    # now First stream can be sent to the first neural net to check the spatial data(shold look for stuff like unnatural lighting, weird textures, etc.)
    spatial_tensor_patch = patch_pipline(pil_image)

    # STEP 3: now we use openCV to extract frequency data from the image
    open_cv_array = np.array(pil_image) #image -->numpy matrix
    gray_image = cv2.cvtColor(open_cv_array, cv2.COLOR_RGB2GRAY) #numpy matrix --> grayscale
    
    #calculate first frontier transform
    f_transform = np.fft.fft2(gray_image)
    f_shift = np.fft.fftshift(f_transform)

    #calculate magnitude spectrum
    frequency_spectrum = 20*np.log(np.abs(f_shift)+ 1e-8)

    #now second stream can be sent to the second neural net
    return spatial_tensor_patch, frequency_spectrum

# spatial_data, frequency_data = image_processor("preprocess_scripts/uwp3723310.jpeg")
# print(spatial_data)
def folder_processor(folder_path):
    


    