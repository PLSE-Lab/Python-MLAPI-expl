import os
import imageio
import numpy as np 
import pandas as pd 

from PIL import Image, ImageOps

def resize_image(srcfile, 
                 destfile, 
                 new_width=128, 
                 new_height=128):
    '''
    Call this function to resize a single image.
    
    srcfile - String, path to the image that will be resized
    destfile - String, path to the location where resized image will be saved
    new_width - Integer
    new_height, Integer
    
    NOTE: set destfile = srcfile if you want to override the original image
    '''
    pil_image = Image.open(srcfile)
    pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
    pil_image_rgb = pil_image.convert('RGB')
    pil_image_rgb.save(destfile, format='JPEG', quality=90)
    return destfile


def dataset_resizer(csv_path, 
                    new_width=128,
                    new_height=128,
                    override=False, 
                    resized_images_folder=None):
    '''
    Resize all images in the dataset.
    
    csv_path - String, path to the main_dataset.csv file
    new_width - Integer
    new_height - Integer
    override - Boolean, if True original image will be overridden by the resized one
    resized_images_folder - String, if override is False, new folder will be created for resized images, this is where you can provide name for that folder
    '''
    
    dataset = pd.read_csv(csv_path)
    
    resized_paths = []
    if override:
    
        for path in dataset.img_paths.values:
            resized_paths.append(resize_image(path, path, new_width, new_height))
        
    else:
        
        if resized_images_folder is not None and not os.path.exists(resized_images_folder):
            os.mkdir(resized_images_folder)
        else:
            resized_images_folder = "resized_images"
            if not os.path.exists(resized_images_folder):
                os.mkdir(resized_images_folder)
            
        for path in dataset.img_paths.values:
            sub_folder = path.split("/")[1]
            if not os.path.exists(resized_images_folder + "/" + sub_folder):
                os.mkdir(resized_images_folder + "/" + sub_folder)
                
            resized_paths.append(resize_image(path, path.replace("dataset", resized_images_folder), new_width, new_height))
            
    return resized_paths
            
    
def load_dataset_to_ram_csv(csv_path):
    '''
    Call this function to load all dataset images into memory.
    
    csv_path, String, path to the main_dataset.csv file
    '''
    
    images = []
    
    dataset = pd.read_csv(csv_path)
    
    for path in dataset.img_paths.values:
        
        images.append(imageio.imread(path, pilmode='RGB'))
        
    return np.array(images)


def load_dataset_to_ram_list(path_list):
    '''
    Call this function to load all dataset images into memory.
    
    path_list, Python list of strings, list of image paths (i.e. This is useful after using dataset_resizer function)
    '''
    
    images = []
    
    for path in path_list:
        
        images.append(imageio.imread(path, pilmode='RGB'))
        
    return np.array(images)