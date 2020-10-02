import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os
#import glob

#create Correlation Matrice for the target column or other column:

def createCorrMat(filePath,target_columnName,LargestColumns):
    data=pd.read_csv(filePath)
    #taget column=the top column in corrMat
    #filename is needed for data
    #LargestColumns=number of column name in the corrMat 
    
    corrmat = data.corr()
    cols = corrmat.nlargest(LargestColumns,target_columnName)[target_columnName].index
    cm = np.corrcoef(data[cols].values.T)
    sns.set(font_scale=1.0)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 6}, yticklabels=cols.values,
                 xticklabels=cols.values)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.show()
    plt.savefig('corrMat.png')


