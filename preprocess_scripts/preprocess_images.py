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
    print("scanning folder...")

    spatial_list = []
    frequency_list = []
    valid_files = []

    search_patterns = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    image_files = []

    for pattern in search_patterns:
        image_files.extend(glob.glob(os.path.join(folder_path, pattern)))
    if not image_files:
        print("no valid images found")
        return None, None
    for img in image_files:
        with open(img, 'rb') as file_stream: #rb means read binary data
            spatial_patch, frequency_spectrum = image_processor(file_stream) # we pass a file stream 
            if spatial_patch is not None and frequency_spectrum is not None:
                spatial_list.append(spatial_patch)

                frequency_tensor = torch.tensor(frequency_spectrum, dtype = torch.float32).unsqueeze(0)
                frequency_list.append(frequency_tensor)
                valid_files.append(os.path.basename(img))
                print(f"processed: {os.path.basename(img)}")
            else:
                print(f"failed: {os.path.basename(img)}")
    
    spatial_batch = torch.stack(spatial_list) #we stack the data as the neural net will be running on gps and should be able to use multiple groups of data to learn
    frequency_batch = torch.stack(frequency_list)

    print("proces complete")
    print(spatial_batch.shape)
    print(frequency_batch.shape)

    return spatial_batch, frequency_batch

spatial_batch, freq_batch = folder_processor("<ENTER FOLDER PATH>")

    
    


    