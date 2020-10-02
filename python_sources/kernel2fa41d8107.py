'''importing libraries'''
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
import pickle

dtypes = {
        'MachineIdentifier':                                    'category',
        'ProductName':                                          'category',
        'EngineVersion':                                        'category',
        'AppVersion':                                           'category',
        'AvSigVersion':                                         'category',
        'IsBeta':                                               'int8',
        'RtpStateBitfield':                                     'float16',
        'IsSxsPassiveMode':                                     'int8',
        'DefaultBrowsersIdentifier':                            'float16',
        'AVProductStatesIdentifier':                            'float32',
        'AVProductsInstalled':                                  'float16',
        'AVProductsEnabled':                                    'float16',
        'HasTpm':                                               'int8',
        'CountryIdentifier':                                    'int16',
        'CityIdentifier':                                       'float32',
        'OrganizationIdentifier':                               'float16',
        'GeoNameIdentifier':                                    'float16',
        'LocaleEnglishNameIdentifier':                          'int8',
        'Platform':                                             'category',
        'Processor':                                            'category',
        'OsVer':                                                'category',
        'OsBuild':                                              'int16',
        'OsSuite':                                              'int16',
        'OsPlatformSubRelease':                                 'category',
        'OsBuildLab':                                           'category',
        'SkuEdition':                                           'category',
        'IsProtected':                                          'float16',
        'AutoSampleOptIn':                                      'int8',
        'PuaMode':                                              'category',
        'SMode':                                                'float16',
        'IeVerIdentifier':                                      'float16',
        'SmartScreen':                                          'category',
        'Firewall':                                             'float16',
        'UacLuaenable':                                         'float32',
        'Census_MDC2FormFactor':                                'category',
        'Census_DeviceFamily':                                  'category',
        'Census_OEMNameIdentifier':                             'float16',
        'Census_OEMModelIdentifier':                            'float32',
        'Census_ProcessorCoreCount':                            'float16',
        'Census_ProcessorManufacturerIdentifier':               'float16',
        'Census_ProcessorModelIdentifier':                      'float16',
        'Census_ProcessorClass':                                'category',
        'Census_PrimaryDiskTotalCapacity':                      'float32',
        'Census_PrimaryDiskTypeName':                           'category',
        'Census_SystemVolumeTotalCapacity':                     'float32',
        'Census_HasOpticalDiskDrive':                           'int8',
        'Census_TotalPhysicalRAM':                              'float32',
        'Census_ChassisTypeName':                               'category',
        'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float16',
        'Census_InternalPrimaryDisplayResolutionHorizontal':    'float16',
        'Census_InternalPrimaryDisplayResolutionVertical':      'float16',
        'Census_PowerPlatformRoleName':                         'category',
        'Census_InternalBatteryType':                           'category',
        'Census_InternalBatteryNumberOfCharges':                'float32',
        'Census_OSVersion':                                     'category',
        'Census_OSArchitecture':                                'category',
        'Census_OSBranch':                                      'category',
        'Census_OSBuildNumber':                                 'int16',
        'Census_OSBuildRevision':                               'int32',
        'Census_OSEdition':                                     'category',
        'Census_OSSkuName':                                     'category',
        'Census_OSInstallTypeName':                             'category',
        'Census_OSInstallLanguageIdentifier':                   'float16',
        'Census_OSUILocaleIdentifier':                          'int16',
        'Census_OSWUAutoUpdateOptionsName':                     'category',
        'Census_IsPortableOperatingSystem':                     'int8',
        'Census_GenuineStateName':                              'category',
        'Census_ActivationChannel':                             'category',
        'Census_IsFlightingInternal':                           'float16',
        'Census_IsFlightsDisabled':                             'float16',
        'Census_FlightRing':                                    'category',
        'Census_ThresholdOptIn':                                'float16',
        'Census_FirmwareManufacturerIdentifier':                'float16',
        'Census_FirmwareVersionIdentifier':                     'float32',
        'Census_IsSecureBootEnabled':                           'int8',
        'Census_IsWIMBootEnabled':                              'float16',
        'Census_IsVirtualDevice':                               'float16',
        'Census_IsTouchEnabled':                                'int8',
        'Census_IsPenCapable':                                  'int8',
        'Census_IsAlwaysOnAlwaysConnectedCapable':              'float16',
        'Wdft_IsGamer':                                         'float16',
        'Wdft_RegionIdentifier':                                'float16',
        'HasDetections':                                        'int8'
        }
        
        
data=pd.read_csv('../input/train.csv',dtype=dtypes)
test=pd.read_csv('../input/test.csv')

data['ProductName'].dtype
(test[test.columns[0]]).dtype

'''plot missinfg value counts'''
(data.isnull().sum()).plot(kind ='pie')
(data.isnull().sum()).plot(kind ='bar')

'''get output variable'''
output=list(set(data.columns)-set(test.columns))[0]

'''check skewness in output'''
data[output].value_counts().plot(kind='pie')

numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
'''get statistic type of columns'''
statistic=list(filter(lambda x: (data[x]).dtype  in  numerics , data.columns ))

'''get catagorical type of  col'''
catagorical=list(filter(lambda x: (data[x]).dtype not in  numerics , data.columns ))

'''verify count of ttal columns'''
print(len(statistic)+len(catagorical)==len(data.columns))

'''missing for stastistical data only'''
stat_missing=data[statistic].isnull().sum()/len(data[statistic[0]])*100

'''Dealing Missing values'''
'''without this you cant go for PCA'''
''' Dropping columns with higher than 75% of missing value'''
'''missing fill by mod value'''
to_drop=(stat_missing.loc[stat_missing>70]).index
new_data=list(set(statistic)-set(to_drop))
pca_data=data[new_data]
pca_data=pca_data.drop(output,axis=1)

for each in pca_data.columns:
    pca_data[each]=pca_data[each].fillna(pca_data[each].mode()[0])





'''plotting heatmap'''
sns.heatmap(data[stat].corr())

'''Apply PCA to reduce dimentionality on continuous data'''
'''here how to select n comonenet value'''
pca = PCA(n_components=5)
principalComponents = pca.fit_transform(pca_data)
principalDf = pd.DataFrame(data = principalComponents , columns = ['pc1', 'pc2','pc3','pc4','pc5'])
principalDf.head(10)

'''correlation for statistical data and output data '''
for each in principalDf.columns:
    print(principalDf[each].corr(data[output]))
for each in pca_data.columns:
    print(pca_data[each].corr(data[output]))


'''catagorical data'''
cat_data=data[catagorical]
cat_data.columns
for each in (cat_data.columns):
    print(each , cat_data[each].nunique())

'''catagorical columns to drop having catagorical value more than 20'''    
drop_catagorical=list(filter(lambda x: cat_data[x].nunique() >20, cat_data.columns))
cat_data=cat_data.drop(drop_catagorical,axis=1)


'''removing columns having occurances of only 1 catagories occurance for more than 90% out of all catagories i.e. skewed data'''
to_drop=list(filter(lambda x: cat_data[x].value_counts().max()/len(cat_data[x])*100 > 90 , cat_data.columns))
cat_data=cat_data.drop(to_drop,axis=1)
cat_data.head(2)

'''Dealing missing value in catagoricals'''
to_drop=list(filter(lambda x: cat_data[x].isnull().sum()/cat_data.shape[0]*100> 70,cat_data.columns))
cat_data=cat_data.drop(to_drop,axis=1)
cat_data.isnull().sum()/cat_data.shape[0]*100

'''filling null in catagorical data with mode value'''
for each in cat_data.columns:
    cat_data[each]=cat_data[each].fillna(cat_data[each].value_counts().index[0])


'''Encoding of catagorical data '''
from sklearn import preprocessing

for each in cat_data.columns:
    le = preprocessing.LabelEncoder()
    try:
        cat_data[each]=le.fit_transform(cat_data[each])
    except:
        print('unable to encode this ' , each)
        
'''Toal final input dataframe and reasult dataframe'''        
final_input_data=pd.concat([principalDf,cat_data],axis=1)
output_data=data[output]

'''train test grouping of data'''
from sklearn.model_selection import train_test_split
final_input_train, final_input_test, output_train, output_test = train_test_split(final_input_data, output_data, test_size=0.33, random_state=42)
 
'''XGB classifier'''
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
model = XGBClassifier()
model.fit(final_input_train, output_train)
print(model) 

''' make predictions for test data'''
output_pred = model.predict(final_input_test)
accuracy_score(output_test,output_pred)


'''Evaluation matrices'''
from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(output_pred, output_test, average='macro')

'''saving the model'''
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))