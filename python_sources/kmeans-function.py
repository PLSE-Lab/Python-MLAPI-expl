
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


def cluster(n, df): 
    '''
    ARGS:   N = NUMBER OF CLUSTERS FOR KMEANS, 
            DF = DATAFRAME OF 1 CATEGORICAL VARIABLE WITH SALES PRICE
            TARGETVAR = VARIABLE TO CLUSTER (E.G. NEIGHBORHOODS ETC..)
    '''      
    xvar = df.columns[0]
    yvar = df.columns[1]

    #MAKE LABELS OF CATEGORICAL VARIABLE BECAUSE K MEANS CAN'T HANDLE STRINGS

    
    lbl = df[xvar].unique()
    lbl = pd.DataFrame(lbl)
    lbl['mylablels']= list(range(1,len(lbl)+1))
    lbl.columns = [xvar, 'mylabels']
    df = df.merge(lbl, on = xvar)
    df = df[["mylabels", yvar]]
    df_means = df.groupby("mylabels").agg("mean").reset_index().sort_values(yvar)
    df_array = df_means.to_numpy()
    
    
    #APPLY K MEANS
    kfit = KMeans(n_clusters=n, random_state=0).fit(df_array)
    klabels = kfit.labels_
    klabels = pd.DataFrame(klabels)
    k_df = pd.DataFrame(df_array)
    k_df['klabels'] = klabels
    colnames = df.columns
    k_df.columns = [colnames[0], yvar, 'klabels']
    
   
    #TRANSFORM TO A DATAFRAME LOOKUP OF LABELS TO ORIGINAL VARIABLE 
    kDict = k_df[['klabels', 'mylabels', yvar]]
    kDict = k_df.set_index('klabels').sort_values(yvar).reset_index()
    kDict = k_df.merge(lbl, on= 'mylabels')[['klabels',xvar, yvar]].sort_values(yvar) #Sale price is mean in each category 
    return kDict


#Example:
#data = pd.read_csv('data_clean.csv')
#train = data[0:1460]

#cluster(2, train[['MSZoning', 'SalePrice']])

#test = cluster(5,train[['Neighborhood', 'SalePrice']])




def kReplace(df, kdf, xvar, yvar): 

    '''
    Replace original column with klabeled column in data 

    ARGS:   DF: ORIGINAL DATA
            KDF: OUTPUT OF CLUSTER FUNCTION
            XVAR: E.G. NEIGHBORHOOD
            YVAR: E.G. SALE PRICE
    '''

    kData = df
    ktargetname = 'k' + xvar
    kData[ktargetname]= ""
    
    kdf = kdf.drop(yvar, axis = 1)
    kdf = kdf.set_index(xvar).transpose().to_dict(orient='list')
   
    for i in np.arange(len(df)):
        kData[ktargetname].loc[i]  = kdf.get(df[xvar].loc[i])
        
    kData = kData.drop(xvar, axis = 1)
    kData = kData.rename(columns={ktargetname: xvar})
    
    for i in np.arange(len(kData)):
        kData[xvar].loc[i] = kData[xvar].loc[i][0]
    
    return kData


#Example:
#print(kReplace(data, test, 'Neighborhood', 'SalePrice'))


