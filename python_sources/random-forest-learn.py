import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def create_nan_dict(data):
    ret = dict()
    for col in data:
        if col != 'HasDetections' and col != 'MachineIdentifier':
            ret[col] = data[col].astype('float32').mean()
    return ret


def remove_nans(data, nan_dict):
    for col in data:
        if col != 'HasDetections' and col != 'MachineIdentifier':
            data[col] = data[col].fillna(nan_dict[col])


# Train the given learning algorithm. Predict the result and extract the output.
print('Loading the training set:')
dtypes = {
    'MachineIdentifier': 'category',
    'AVProductsInstalled': 'float32',
    'CountryIdentifier': 'float32',
    'OrganizationIdentifier': 'float32',
    'GeoNameIdentifier': 'float32',
    'LocaleEnglishNameIdentifier': 'float32',
    'OsBuild': 'int16',
    'OsSuite': 'float32',
    'OsPlatformSubRelease': 'float32',
    'SkuEdition': 'float32',
    'IeVerIdentifier': 'float32',
    'SmartScreen': 'float32',
    'Census_MDC2FormFactor': 'float32',
    'Census_ProcessorCoreCount': 'float16',
    'Census_ProcessorManufacturerIdentifier': 'float32',
    'Census_PrimaryDiskTotalCapacity': 'float32',
    'Census_PrimaryDiskTypeName': 'float32',
    'Census_SystemVolumeTotalCapacity': 'float32',
    'Census_TotalPhysicalRAM': 'float32',
    'Census_ChassisTypeName': 'float32',
    'Census_InternalPrimaryDiagonalDisplaySizeInInches': 'float16',
    'Census_InternalPrimaryDisplayResolutionHorizontal': 'float16',
    'Census_InternalPrimaryDisplayResolutionVertical': 'float16',
    'Census_PowerPlatformRoleName': 'float32',
    'Census_InternalBatteryType': 'float32',
    'Census_InternalBatteryNumberOfCharges': 'float32',
    'Census_OSBranch': 'float32',
    'Census_OSBuildNumber': 'int16',
    'Census_OSBuildRevision': 'int32',
    'Census_OSEdition': 'float32',
    'Census_OSSkuName': 'float32',
    'Census_OSInstallTypeName': 'float32',
    'Census_OSInstallLanguageIdentifier': 'float32',
    'Census_OSUILocaleIdentifier': 'float32',
    'Census_OSWUAutoUpdateOptionsName': 'float32',
    'Census_GenuineStateName': 'float32',
    'Census_ActivationChannel': 'float32',
    'Census_IsFlightingInternal': 'float32',
    'Census_ThresholdOptIn': 'float16',
    'Census_IsSecureBootEnabled': 'int8',
    'Census_IsWIMBootEnabled': 'float32',
    'Census_IsTouchEnabled': 'int8',
    'Wdft_IsGamer': 'float32',
    'Wdft_RegionIdentifier': 'float32',
    'HasDetections': 'int8',
    'EngineVersion_0': 'float32',
    'EngineVersion_1': 'float32',
    'EngineVersion_2': 'float32',
    'EngineVersion_3': 'float32',
    'AppVersion_0': 'float32',
    'AppVersion_1': 'float32',
    'AppVersion_2': 'float32',
    'AppVersion_3': 'float32',
    'AvSigVersion_0': 'float32',
    'AvSigVersion_1': 'float32',
    'AvSigVersion_2': 'float32',
    'AvSigVersion_3': 'float32',
    'Census_OSVersion_0': 'float32',
    'Census_OSVersion_1': 'float32',
    'Census_OSVersion_2': 'float32',
    'Census_OSVersion_3': 'float32'
}

# Read the data set
training_set = pd.read_csv('../input/prepare-data-for-decision-trees-algorithms/training_decisionTrees.csv',
                           dtype=dtypes)
print('Training set loaded')
print(training_set.shape)

# Handle NaNs
print('Handling NaN in training set')
nan_to_num_dict = create_nan_dict(training_set)
remove_nans(training_set, nan_to_num_dict)
print('Done')


# Now we remove columns that have only one value. Those columns are not relevant for Decision Tree and can be removed
# for run time
columns_to_remove = []
for col_name in training_set.columns.values:
    if col_name == 'HasDetections' or col_name == 'MachineIdentifier':
        continue
    unique_values = training_set[col_name].value_counts(dropna=False)
    if len(unique_values) == 1:
        del training_set[col_name]
        columns_to_remove.append(col_name)

print(str(len(columns_to_remove)) + ' columns removed')
print(training_set.shape)

# Prepare the learning
features = [c for c in training_set.columns if c not in ['MachineIdentifier', 'HasDetections']]
x_train = training_set[features]
y_train = training_set['HasDetections']
del training_set

rand_forest = RandomForestClassifier(n_estimators=200, min_samples_leaf=20, n_jobs=-1)
print('learning')
rand_forest.fit(x_train, y_train)
print('Done')
del x_train, y_train

# Read the test set and parse it
test = pd.read_csv('../input/prepare-data-for-decision-trees-algorithms/test_decisionTrees.csv', dtype=dtypes)
remove_nans(test, nan_to_num_dict)
x_test = test[features]
to_submit = pd.DataFrame(test['MachineIdentifier'])
del test
y_pred = rand_forest.predict_proba(x_test)[:, 1]
to_submit['HasDetections'] = y_pred
to_submit.to_csv('randomForest_minLeaf_10.csv', index=False)