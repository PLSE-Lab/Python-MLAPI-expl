# Import necessary modules
from timeit import default_timer as timer
from sklearn.decomposition import PCA
import pandas as pd 
import numpy as np

#### Dimensionalty reduction using PCA/LDA ####
def Reduce_images(Images,Technique,components):
    
    if Technique =="PCA": # PRINCIPAL COMPONENT ANALYSIS
        pca = PCA(n_components=components)
        pca.fit(Images)
        transformed_images= pca.transform(Images)
        
    else : # BOTH PCA & LDA (TODO)
        transformed_images= Images
    
    print("------IMAGES REDUCED SUCCESSFULLY-----")
    print("Data reduced from "+str(Images.shape)+" to "+str(transformed_images.shape)+" with variance loss of "
    +str(1-np.sum(pca.explained_variance_ratio_)))    
        
    return transformed_images
    

if __name__ == "__main__":
    start= timer()
    labeled_images = pd.read_csv('../input/train.csv')
    images = labeled_images.iloc[:,1:]
    labels = labeled_images.iloc[:,:1]
    X= Reduce_images(images,Technique="PCA",components=400)
    y=labels
    end= timer()
    print("Time taken:"+str(end-start))