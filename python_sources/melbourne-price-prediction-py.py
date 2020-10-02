# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#!/usr/bin/python
import matplotlib
#matplotlib.use('Agg')
import pandas as pd
import numpy as np
from tensorflow.contrib import keras
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import argparse
import os

def string2integer(ref,data,key):
    if key in ref:
        select = data[key]
        integer_encoding = LabelEncoder().fit_transform(select)
        data[key] = integer_encoding


def guess_regionname(data,target_suburb,target_street):
    ref = data['Regionname'].isnull()
    # if there is entry within same subrn then return its region if available
    for i in range(0,len(data)):
        suburb = data['Suburb'][i]
        if target_suburb == suburb and ref[i] == False:
            return data['Regionname'][i]  
        
    # if there is entry within street name then return its RegionName
    for i in range(0,len(data)):
        street = data['Address'].values[i][data['Address'].values[i].find(" ")+1:]
        if street == target_street and ref[i] == False:
            return data['Regionname'][i]  
    return "Unknown"


def fill_region_name_up(data):
    ref = data['Regionname'].isnull()
    for i in range(0,len(ref)):
        if ref[i] == True:
            street = data['Address'].values[i][data['Address'].values[i].find(" ")+1:]
            suburb = data['Suburb'][i]
            data['Regionname'][i] = guess_regionname(data,suburb,street)
    return


def guess_property_count(data,target_suburb):
    ref = data['Propertycount'].isnull()
    for i in range(0,len(data)):
        if target_suburb == data['Suburb'][i] and ref[i] == False:
            return data['Propertycount'][i]
    return 1 

def fill_property_count_up(data):
    """ Idea is to find property count from other entries of the same suburb"""
    ref = data['Propertycount'].isnull()
    for i in range(0,len(ref)):
        if ref[i] == True:
            data['Propertycount'][i] = guess_property_count(data,data['Suburb'][i]) 
    return


def guess_distance(data, target_suburb):
    ref = data['Distance'].isnull()
    sum_distance = 0.0
    sum_counter = 0.0
    for i in range(0,len(data)):
         if ref[i] == False and data['Suburb'][i] == target_suburb:
            sum_distance += data['Distance'][i]
            sum_counter += 1.0
    if sum_counter == 0:
        sum_counter = 1.0
    return sum_distance/sum_counter


def fill_distance_up(data):
    """ Distance to C.B.D taken as an averge distance from other properties in
        the same suburb located on the same street. If no properties on the same street
        then averge distance in the same suburb"""
    ref = data['Distance'].isnull()
    to_drop = []
    for i in range(0,len(ref)):
        if ref[i] == True:
            data['Distance'][i] = guess_distance(data,data['Suburb'][i]) 
            # If in ths suburb there is only one property with borken data
            # then delete this entry
            if data['Distance'][i] == 0:
                to_drop.append(i)
    return data.drop(to_drop)
    

def guess_yearbuilt(data, target_suburb, target_type):
    ref = data['YearBuilt'].isnull()
    sum_yearbuilt = 0.0
    sum_counter = 0.0
    for i in range(0,len(data)):
         if (ref[i] == False) and (data['Suburb'][i] == target_suburb) and (data['Type'][i] == target_type):
            sum_yearbuilt += data['YearBuilt'][i]
            sum_counter += 1.0
    if sum_counter == 0:
        sum_counter = 1.0
    return sum_yearbuilt/sum_counter


def fill_yearbuilt_up(data):
    """ If year build is missing find same type property from same suburb and takes its year 
    provided it is no fresher that Date of transaction"""
    ref = data['YearBuilt'].isnull()
    for i in range(0,len(ref)):
        if ref[i] == True:
            data['YearBuilt'][i] = guess_yearbuilt(data,data['Suburb'][i],data['Type'][i])
            print("Type: %s Suburb: %s Guessed year: %d" %(data['Type'][i],data['Suburb'][i],data['YearBuilt'][i]))
    return 

def guess_value_by_interpolation(ref,data,target_suburb,target_type,target_key,target_val, gap_key, properties_stats):
    if target_val == pd.np.nan or target_val == 0:
        return 0
    neighbour_less_stats = properties_stats[(properties_stats['Suburb'] == target_suburb) &
            (properties_stats[target_key] <= target_val)]
    neighbour_greater_stats = properties_stats[(properties_stats['Suburb'] == target_suburb) &
            (properties_stats[target_key] > target_val)]
    if len(neighbour_less_stats) != 0 and len(neighbour_greater_stats) != 0:
        min_index = neighbour_less_stats[target_key].argmax()
        max_index = neighbour_greater_stats[target_key].argmin()
        delta_y = neighbour_greater_stats[gap_key][max_index] - neighbour_less_stats[gap_key][min_index] 
        delta_x = neighbour_greater_stats[target_key][max_index] - neighbour_less_stats[target_key][min_index] 
        estimate = (target_val - neighbour_less_stats[target_key][min_index]) * delta_y/delta_x + neighbour_less_stats[gap_key][min_index]
    elif len(neighbour_less_stats) != 0:
        min_index = neighbour_less_stats[target_key].argmax()
        estimate = neighbour_less_stats[gap_key][min_index]
    elif len(neighbour_greater_stats) !=0:
        max_index = neighbour_greater_stats[target_key].argmin()
        estimate = neighbour_greater_stats[gap_key][max_index]
    else:
        estimate = 0
    return estimate 

def fill_buildingarea_up(data):
    """ Estimate BuildingArea based on type of property and LandSize is available """

    ref = data['BuildingArea'].isnull()
    # Make a hash table ['type' : data]
    properties_stats = {}
    properties_stats['h'] = data[(data['Type'] == "h") & (data['Landsize'].notnull())
            & (data['BuildingArea'].notnull())]
    properties_stats['t'] = data[(data['Type'] == "t") & (data['Landsize'].notnull())
            & (data['BuildingArea'].notnull())]
    properties_stats['br'] = data[(data['Type'] == "br") & (data['Landsize'].notnull())
            & (data['BuildingArea'].notnull())]
    properties_stats['u'] = data[(data['Type'] == "u") & (data['Landsize'].notnull())
            & (data['BuildingArea'].notnull())]
    properties_stats['dev site'] = data[(data['Type'] == "dev site") & (data['Landsize'].notnull())
            & (data['BuildingArea'].notnull())]
    properties_stats['o res'] = data[(data['Type'] == "o res") & (data['Landsize'].notnull())
            & (data['BuildingArea'].notnull())]
    
    for i in range(0,len(ref)):
        if ref[i] == True:
            data['BuildingArea'][i] = guess_value_by_interpolation(ref,data,data['Suburb'][i],data['Type'][i],
                    'Landsize',data['Landsize'][i], 'BuildingArea', properties_stats[data['Type'][i]])
    return data


def fill_landsize_up(data):
    """ Estimate LAndsize based on type of property and BuildingArea if available """

    ref = data['Landsize'].isnull()
    # Make a hash table ['type' : data]
    properties_stats = {}
    properties_stats['h'] = data[(data['Type'] == "h") & (data['BuildingArea'].notnull())
            & (data['Landsize'].notnull())]
    properties_stats['t'] = data[(data['Type'] == "t") & (data['BuildingArea'].notnull())
            & (data['Landsize'].notnull())]
    properties_stats['br'] = data[(data['Type'] == "br") & (data['BuildingArea'].notnull())
            & (data['Landsize'].notnull())]
    properties_stats['u'] = data[(data['Type'] == "u") & (data['BuildingArea'].notnull())
            & (data['Landsize'].notnull())]
    properties_stats['dev site'] = data[(data['Type'] == "dev site") & (data['BuildingArea'].notnull())
            & (data['Landsize'].notnull())]
    properties_stats['o res'] = data[(data['Type'] == "o res") & (data['BuildingArea'].notnull())
            & (data['Landsize'].notnull())]

    for i in range(0,len(ref)):
        if ref[i] == True:  # guess building area
            data['Landsize'][i] = guess_value_by_interpolation(ref,data,data['Suburb'][i],data['Type'][i],
                    'BuildingArea',data['BuildingArea'][i], 'Landsize', properties_stats[data['Type'][i]])
    return data

def load_and_preprocess_data(melbourne_file_path):
    melbourne_data = pd.read_csv(melbourne_file_path)
    print(melbourne_data.isnull().sum()) # This is printing missing data
    # Car: number of parking spots. NAN will be replaced wth zeros
    melbourne_data['Car'].fillna(0,inplace=True)
    # Bathroom: number of bathrroms. NAN will be replaced with ones
    melbourne_data['Bathroom'].fillna(1,inplace=True)
    # Try to make up the field of council area
    melbourne_data['CouncilArea'].fillna("Unknown",inplace=True)
    # Try to fill the missing RegionName
    fill_region_name_up(melbourne_data)
    # Try to fill the missing PropertyCount (properites count in suburb)
    fill_property_count_up(melbourne_data)
    # Try to guess/even out YearBuild
    melbourne_data['YearBuilt'].fillna(0,inplace=True)
    # Building Area estimation 
    melbourne_data = fill_buildingarea_up(melbourne_data)
    # Landsize estimation
    melbourne_data = fill_landsize_up(melbourne_data)

    # Distance to C.B.D to be made up based on street+suburb info
    melbourne_data = fill_distance_up(melbourne_data)

    melbourne_data = melbourne_data.dropna(axis=0, subset = ['Price'])  # Drop data that contains NAN in Price column
    print(melbourne_data.isnull().sum()) # This is printing missing data
    if args.preprocess_only == True:
        melbourne_data.to_csv("preprocessed.csv")
        #melbourne_data.to_csv(args.dataset[5:].strip('.csv')+"_preprocessed.csv")
        return melbourne_data
    ref = melbourne_data.tail().to_dict()
    # Replace string values of CouncilArea with onehott representation
    string2integer(ref,melbourne_data,'CouncilArea')
    # Replace suburb name with integer encoding value
    string2integer(ref,melbourne_data,'Suburb')
    # Type of property
    string2integer(ref,melbourne_data,'Type')
    # enocode regionname 
    string2integer(ref,melbourne_data,'Regionname')
    # enocode SellerG 
    string2integer(ref,melbourne_data,'SellerG')
    # enocode Method 
    string2integer(ref,melbourne_data,'Method')
    # preprocess adress, remove apartmetn number , leaving just street
    if 'Address' in ref:
        address =  melbourne_data['Address']
        for i in range(0,len(address.values)):
            address.values[i] = address.values[i][address.values[i].find(" ")+1:]
        integer_encoding = LabelEncoder().fit_transform(address)
        melbourne_data['Address'] = integer_encoding

    # Day + month*32 + 32*12*year
    if 'Date' in ref:
        datas = melbourne_data['Date']
        for i in range(0,len(datas.values)):
            splits = datas.values[i].split('/')
            datas.values[i] = int(splits[0]) + int(splits[1])*32 + int(splits[2])*32*12
    return melbourne_data

def prepare_melbourne_dataset(melbourne_dataset_file):
    melbourne_data = load_and_preprocess_data(melbourne_dataset_file)
    # Ignored : Bedroom2, Latitude, Longtitude
    input_features = ['Address','Bathroom','BuildingArea','Car','CouncilArea', 'Date', 'Distance', 'Landsize', 'Method', 'Postcode', 'Price', 'Propertycount', 'Regionname', 'Rooms', 'SellerG', 'Suburb', 'Type', 'YearBuilt']
    return train_test_split(melbourne_data[input_features],melbourne_data.Price)

def prepare_competition_dataset(train_data_file, test_data_file):
    train_data = load_and_preprocess_data(train_data_file)
    test_data = load_and_preprocess_data(test_data_file)
    # Ignored:
    # Alley -- Almost all NAN
    input_features = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'Bedroom', 'Kitchen', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition']

    return train_data,train_data['SalePrice'],test_data,test_data['SalePrice'] 

def parse_desc(model_desc):
    ''' Parse desc of model. Format is num of units in first layer, num of units in next layer..
        num of unit is last layer(has to be one)'''
    desc = []
    splitted = model_desc.split(',')
    for item in splitted:
        desc.append(int(item))
    if desc[len(desc)-1] != 1:
      print("Error: final layer should have one unit!")
      exit(-1)
    return desc


def make_model(model_name,num_features):
    # Building model: 
    adam = keras.optimizers.Adam() # Does converge slowly 
    melbourne_model = keras.models.Sequential(name="-"+model_name + "-Adam")
    
    melbourne_model.add(keras.layers.Dense(20, activation='tanh',input_dim=num_features))
    melbourne_model.add(keras.layers.Dense(20, activation='tanh'))
    melbourne_model.add(keras.layers.Dense(1, activation='relu', kernel_initializer='he_normal'))   # MAE: 922165

    #sgd = keras.optimizers.SGD(lr=0.0005, decay=1e-6, momentum=0.9) # MAE: 460326
 #   melbourne_model.compile(loss="mean_squared_error", optimizer='rmsprop') # Does converge slowly
    melbourne_model.compile(loss="mean_squared_error", optimizer=adam) # Does converge slowly
    return melbourne_model

def make_relu_model(model_name,model_desc,num_features):

    # Prase description of model
    desc = parse_desc(model_desc)

    # Building model: 
    adam = keras.optimizers.Adam() # Does converge slowly 
    melbourne_model = keras.models.Sequential(name="-"+model_name + "-Adam")
    
    desc_str = "Relu model: "
    for i in range(0,len(desc)):
        if i == 0:
            melbourne_model.add(keras.layers.Dense(20, activation='relu', kernel_initializer='he_normal', input_dim=num_features))
            desc_str += str(desc[i])
        else:
            melbourne_model.add(keras.layers.Dense(desc[i], activation='relu', kernel_initializer='he_normal'))
            desc_str += "-"+str(desc[i])

    print(desc_str)
    melbourne_model.compile(loss="mean_squared_error", optimizer=adam)
    return melbourne_model

# TODO: figure out how to initialize bias for SNN
def make_selu_model(model_name,model_desc,num_features):

    # Prase description of model
    desc = parse_desc(model_desc)

    desc_str = "SNN model: "
    melbourne_model = keras.models.Sequential(name="-"+model_name+"-Adam")

    for i in range(0,len(desc)):
        if i == 0:
            melbourne_model.add(keras.layers.Dense(20, kernel_initializer='lecun_normal',bias_initializer='lecun_normal',
                activation='selu',input_dim=num_features))
            desc_str += str(desc[i])
        else:
            melbourne_model.add(keras.layers.Dense(1, kernel_initializer='lecun_normal', bias_initializer='lecun_normal',
                activation='selu'))   # MAE: 
            desc_str += "-"+str(desc[i])

    print(desc_str)

    adam = keras.optimizers.Adam() # Does converge slowly 
    #sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9) # MAE: 
    melbourne_model.compile(loss="mean_squared_error", optimizer=adam) # Does converge slowly
    return melbourne_model

def normalize_input(data,features):
    """ Modify data so it is zero meaned """
    # Get all data from selected columns across samples
    # TODO: remove warnings
    for col in features:
        scaler = StandardScaler().fit(data[col])
        data[col] = scaler.transform(data[col])

def train(model_name, model_desc, num_epochs, X, y):

    initial_epoch = 1
    if model_name == "" or model_name == "FFN": 
        model_name = "FFN"
        melbourne_model = make_relu_model(model_name,model_desc,len(X.columns))
    elif model_name == "SNN":
        normalize_input(X,input_features)
        melbourne_model = make_selu_model(model_name,model_desc,len(X.columns))
    else:
        tmpstr = model_name[:model_name.find("-val_loss")]
        initial_epoch = int(tmpstr[tmpstr.rfind("-")+1:])
        melbourne_model = keras.models.load_model(model_name)
    num_epochs += initial_epoch - 1

    # TODO: Get name of model from compiled model and make
    # name of file to save to based on that
    output = model_name + "_" + model_desc + "-num_epochs-" + str(num_epochs)
    os.mkdir(output)
    show_stopper = keras.callbacks.EarlyStopping(monitor='val_loss',patience=num_epochs-10, verbose=1)
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=output+"/melbourne_model"+melbourne_model.name+".epoch-{epoch:02d}-val_loss-{val_loss:.4f}.hdf5",monitor='val_loss',save_best_only=True,verbose=1)

    history = melbourne_model.fit(X.values, y.values, validation_split=0.2, epochs=num_epochs, initial_epoch=initial_epoch, batch_size=1,callbacks=[show_stopper,checkpoint])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("Loss/cost chart")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.legend(["train","val"],loc="upper right")
    plt.show()
    plt.savefig(output+"/melbourne_model"+melbourne_model.name+"-epochs-"+str(num_epochs))
    print("Making predictions for following houses")
    predictions = melbourne_model.predict(X.values)

    print("MAE:",mean_absolute_error(predictions,y.values))
# The Melbourne data has somemissing values (some houses for which some variables weren't recorded.)
# We'll learn to handle missing values in a later tutorial.  
# Your Iowa data doesn't have missing values in the predictors you use. 
# So we will take the simplest option for now, and drop those houses from our data. 
#Don't worry about this much for now, though the code is:

def infer(model_name, X, y):
    if model_name == "":
        print("Error: Inference mode require model given with --model option")
        exit(-1)


    #melbourne_model.add(keras.layers.Dense(1, activation='relu',input_dim=len(input_features))) # MAE: 1072223
    melbourne_model = make_relu_model(model_name,len(input_features))
    melbourne_model.load_weights(model_name)
    predictions = melbourne_model.predict(X.values)
    print("MAE:",mean_absolute_error(predictions,y.values))
        
    scores = melbourne_model.evaluate(X.values,y.values,verbose=0)
    print("%s: %.2f" % (melbourne_model.metrics_names[0], scores))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Perform training", action="store_true")
    parser.add_argument("--preprocess_only", help="Perform data cleaning and store results in a file", action="store_true")
    parser.add_argument("--infer", help="Perform evaluation", action="store_true")
    parser.add_argument("--type", help="Type of Model to be used for training/inference", type=str, default="")
    parser.add_argument("--model", help="Model to be used for training/inference", type=str, default="13,20,1") # MAE ~16
    #parser.add_argument("--dataset", help="Data Set for training/inference", type=str, default="melb:../input/Melbourne_housing_FULL.csv")
    parser.add_argument("--dataset", help="Data Set for training/inference", type=str, default="melb:./preprocessed.csv")    
    parser.add_argument("--num_epochs", help="Number of epochs to perform", type=int, default=200)
    args = parser.parse_args()

    #args.preprocess_only = True # To generate preprocessed data

    if args.dataset[0:4] == 'melb':
        trainX, testX, trainY, testY = prepare_melbourne_dataset(args.dataset[5:])
    elif args.dataset == 'comp':
        trainX, testX, trainY, testY = prepare_competition_dataset('./train.csv','./test.csv')
    else:
        printf("Invalid value of dataset. Accepted values: 'comp' and 'melb'");

    if args.preprocess_only == True:
        pass
    elif args.infer == True:
        infer(args.model, testX, testY)
    else:
        train(args.type,args.model,args.num_epochs,trainX, trainY)
