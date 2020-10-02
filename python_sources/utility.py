# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing 
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.'
def getData(type="train"):
    if type=="train":
        data=pd.read_csv( '/kaggle/input/sample-linear-regression/training_data_.csv' )
        data.rename(columns=lambda x: x.strip(), inplace=True)
        return data
    elif type=="test":
        data= pd.read_csv( '/kaggle/input/sample-linear-regression/test_data.csv' )
        data.rename(columns=lambda x: x.strip(), inplace=True)
        return data
    else:
        pass
    
def getTrainandTestData(data,properties):
    #print (data)
    X = data.drop(properties['targetColumn'],axis=1) 
    y = data[properties['targetColumn']] #X = df_final.iloc[:,:-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test
    
def getScaledData(data,properties,input_scaler,output_scaler,full_scaler):
    X_train, X_test, y_train, y_test=getTrainandTestData(data,properties)
    # scale inputs
    if input_scaler is not None:
        # fit scaler
        input_scaler.fit(X_train)
        # transform training dataset
        X_train = input_scaler.transform(X_train)
        # transform test dataset
        X_test = input_scaler.transform(X_test)
    if output_scaler is not None:
        # reshape 1d arrays to 2d arrays
        y_train = y_train.values.reshape(len(y_train.values), 1)
        y_test = y_test.values.reshape(len(y_test.values), 1)
        # fit scaler on training dataset
        output_scaler.fit(y_train)
        # transform training dataset
        y_train = output_scaler.transform(y_train)
        # transform test dataset
        y_test = output_scaler.transform(y_test)
    
    if full_scaler is not None:
        full_scaler.fit(data)
        return pd.DataFrame(full_scaler.transform(data),columns=data.columns)
    else:
        return X_train, X_test, y_train, y_test
    
#https://www.dasca.org/world-of-big-data/article/identifying-and-removing-outliers-using-python-packages
#https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba
def remove_outlier(df):
    low = .05
    high = .95
    quant_df = df.quantile([low, high])
    for name in list(df.columns):
      if is_numeric_dtype(df[name]):
       df = df[(df[name] > quant_df.loc[low, name]) 
           & (df[name] < quant_df.loc[high, name])]
    return df


#https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
#https://www.kaggle.com/reisel/how-to-handle-correlated-features
def getReleventFeatures(df,properties,featureSelectionType="Filter Method",featureSelectionSubType="Mutual Correlation",withScaling=False,plotting=True):
    
    #print(df)
    if withScaling:
        X_train, X_test, y_train, y_test=getScaledData(df,properties,properties['scaler'],properties['scaler'],None)
        df=getScaledData(df,properties,None,None,properties['scaler'])
    else:
        X_train, X_test, y_train, y_test=getScaledData(df,properties,None,None,None)
        df=getScaledData(df,properties,None,None,None)
    #print(X_train)
    if featureSelectionType=="Filter Method":
        if 'corr' not in properties:
            cor = df.corr()
        else:
            cor = properties.corr
        #Correlation with output variable
        cor_target = abs(cor[properties['targetColumn']])
        
        if featureSelectionSubType=="Correlation with target":
            #Selecting highly correlated features with target
            relevant_features = cor_target[cor_target>0.5].index.tolist()
        elif featureSelectionSubType=="Mutual Correlation without target" :
            #https://towardsdatascience.com/feature-selection-in-python-recursive-feature-elimination-19f1c39b8d15
            correlated_features = set()
            correlation_matrix = df.drop(properties['targetColumn'], axis=1).corr()

            for i in range(len(correlation_matrix.columns)):
                for j in range(i):
                    if abs(correlation_matrix.iloc[i, j]) > 0.8:
                        colname = correlation_matrix.columns[i]
                        correlated_features.add(colname)
            relevant_features=correlated_features
        elif featureSelectionSubType=="Mutual Correlation" :
            relevant_features_list_mutual=[]
            columns=df.columns
            highlycorrelatecolumns=[]
            #print(cor)
            #One of the assumptions of linear regression is that the independent variables need to be uncorrelated with each other.
            for column1 in columns:
                if column1 not in highlycorrelatecolumns:
                    column1_cor_target = abs(cor[column1])
                    #print("------------correlation of "+column1+"  with value greter then 0.7-------------")
                    highlycorrelatecolumnsForColumn1=column1_cor_target[column1_cor_target>0.7]
                    highlycorrelatecolumnsForColumn1List=highlycorrelatecolumnsForColumn1.index.tolist()
                    #print(highlycorrelatecolumnsForColumn1List)
                    #print(cor_target[highlycorrelatecolumnsForColumn1List])
                    #print(cor_target[highlycorrelatecolumnsForColumn1List].max())

                    maxCorrelatedAmongMutualhighlyCorrelatedColumn=cor_target[highlycorrelatecolumnsForColumn1List].idxmax()
                    relevant_features_list_mutual.append(maxCorrelatedAmongMutualhighlyCorrelatedColumn)
                    #print(maxCorrelatedAmongMutualhighlyCorrelatedColumn)
                    highlycorrelatecolumnsForColumn1List.remove(maxCorrelatedAmongMutualhighlyCorrelatedColumn)
                    #print(highlycorrelatecolumnsForColumn1List)
                    highlycorrelatecolumns+=highlycorrelatecolumnsForColumn1List
                    #print("Final List:")
                    #print( highlycorrelatecolumns)
            relevant_features=relevant_features_list_mutual
            #print(cor_target)
        return relevant_features
    elif featureSelectionType=="Wrapper Method":
        #Backward Elimination
        if featureSelectionSubType=="Backward Elimination":
            #https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
            cols = list(X_train.columns)
            pmax = 1
            while (len(cols)>0):
                p= []
                X_train_1 = X_train[cols]
                X_train_1 = sm.add_constant(X_train_1)
                model = sm.OLS(y_train,X_train_1).fit()
                #print(model.pvalues)
                p = pd.Series(model.pvalues.values[1:],index = cols)      
                pmax = max(p)
                feature_with_p_max = p.idxmax()
                if(pmax>0.05):
                    cols.remove(feature_with_p_max)
                else:
                    break
            selected_features_BE = cols
            return selected_features_BE
        elif featureSelectionSubType=="Recursive Feature Elimination":
            #https://towardsdatascience.com/feature-selection-in-python-recursive-feature-elimination-19f1c39b8d15
            rfc = RandomForestClassifier(random_state=101)
            svc = SVC(kernel="linear", C=1)
            #rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2), scoring='accuracy')
            #rfecv.fit(X_train, y_train)
            rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
            rfe.fit(X_train, y_train)
            #print('Optimal number of features: {}'.format(rfecv.n_features_))
            if ploting:
                plt.figure(figsize=(16, 9))
                plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=18, fontweight='bold', pad=20)
                plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
                plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)
                plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_, color='#303F9F', linewidth=3)

                plt.show()
                
                dset = pd.DataFrame()
                dset['attr'] = X_train.columns
                dset['importance'] = rfe.estimator_.feature_importances_

                dset = dset.sort_values(by='importance', ascending=False)


                plt.figure(figsize=(16, 14))
                plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')
                plt.title('RFECV - Feature Importances', fontsize=20, fontweight='bold', pad=20)
                plt.xlabel('Importance', fontsize=14, labelpad=20)
                plt.show()
                
            return np.where(rfe.support_ == False)[0]
    elif featureSelectionType=="Embedded Method":
        if featureSelectionSubType=="LassoCV":
            #https://www.kaggle.com/sz8416/6-ways-for-feature-selection
            embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l1"), '1.25*median')
            embeded_lr_selector.fit(X_train, y_train)
            embeded_lr_support = embeded_lr_selector.get_support()
            embeded_lr_feature = X_train.loc[:,embeded_lr_support].columns.tolist()
            return embeded_lr_feature
        elif featureSelectionSubType=="LGBMClassifier":
            #https://www.kaggle.com/sz8416/6-ways-for-feature-selection
            lgbc=LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
            reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)

            embeded_lgb_selector = SelectFromModel(lgbc, threshold='1.25*median')
            embeded_lgb_selector.fit(X_train, y_train)
            embeded_lgb_support = embeded_lgb_selector.get_support()
            embeded_lgb_feature = X_train.loc[:,embeded_lgb_support].columns.tolist()
            return embeded_lgb_feature
    else:
        pass
    
#relevant_features=getReleventFeatures(getData(),{'targetColumn':'shares','scaler':preprocessing.RobustScaler()},"Filter Method","Mutual Correlation",True)
#relevant_features=getReleventFeatures(getData(),{'targetColumn':'shares','scaler':preprocessing.RobustScaler()},"Wrapper Method","Backward Elimination",False)
#relevant_features=getReleventFeatures(getData(),{'targetColumn':'shares','scaler':preprocessing.RobustScaler()},"Wrapper Method","Recursive Feature Elimination",False)
#relevant_features=getReleventFeatures(getData(),{'targetColumn':'shares','scaler':preprocessing.RobustScaler()},"Embedded Method","LGBMClassifier",True)
#print("Exported")
#print(relevant_features)

    
    
    
    