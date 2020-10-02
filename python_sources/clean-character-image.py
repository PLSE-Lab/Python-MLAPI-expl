# Import necessary modules
from timeit import default_timer as timer
import pandas as pd 
import numpy as np


def Clean_cimage(dirty_image):
    
    #### NAIVE BINARIZATION ####
    dirty_image[dirty_image>0]=1  
    binary_image= dirty_image
    
    #### REMOVAL OF USELESS COLUMNS ####
    isValid=[np.sum(binary_image.loc[:,x])>0 for x in binary_image.columns]
    Cleaned_image= binary_image.loc[:,isValid]
    
    #### COLUMN REDUNDANCY REMOVAL ####
    Cleaned_image.T.drop_duplicates(keep="first").T
    
    #Cleaned_image.columns.difference(binary_image.columns)
    print("------IMAGE CLEANED SUCCESSFULLY-----")
    print("Data cleaned and reduced from "+str(dirty_image.shape)+" to "+str(Cleaned_image.shape))
    return Cleaned_image
    

if __name__ == "__main__":
    start= timer()
    labeled_images = pd.read_csv('../input/train.csv')
    images = labeled_images.iloc[:,1:]
    labels = labeled_images.iloc[:,:1]
    X= Clean_cimage(images)
    y=labels
    end= timer()
    print("Time taken:"+str(end-start))







