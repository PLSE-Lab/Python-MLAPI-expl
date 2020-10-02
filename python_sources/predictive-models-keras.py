#=======================================================================
#Using Keras with tensorflow backend, and other imports
#=======================================================================
import sys, csv
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import regularizers
from keras.utils.np_utils import to_categorical
from sklearn.svm import SVR

#=======================================================================
#Import training and test data from the csv files
#=======================================================================

debug = 1

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def Regression_DistrictCrimes():
    Train_Data_List = []
    Train_Target_List = []

    with	open('../input/crime_by_district_rt.csv',	'r')	as	f:
        reader	=	csv.DictReader(f,	delimiter=',')
        for	row	in	reader:
            Murder	=	float(row["Murder"])
            Assault_on_women    =   float(row["Assault on women"])
            Kidnapping_and_Abduction  =	float(row["Kidnapping and Abduction"])
            Dacoity	=	float(row["Dacoity"])
            Robbery	=	float(row["Robbery"])
            Arson	=	float(row["Arson"])
            Hurt	=	float(row["Hurt"])
            POA	    =	float(row["Prevention of atrocities (POA) Act"])
            PCR	    =	float(row["Protection of Civil Rights (PCR) Act"])
            Other	=	float(row["Other Crimes Against SCs"])
        
            Train_Data_List.append([Murder, Assault_on_women, Kidnapping_and_Abduction, Dacoity, Robbery, Arson, Hurt, POA, PCR])
            Train_Target_List.append(Other)

    if debug != 0:
        print("=======================Training data=======================")
        print(Train_Data_List)
        print("=========================Test data=========================")
        print(Train_Target_List)


    #=======================================================================
    #Convert the data into 2D Tensors
    #=======================================================================

    train_data =  np.array(Train_Data_List);
    train_targets = np.array(Train_Target_List);

    if debug != 0:
        print("===========================================================")
        print("========================2D tensors=========================")
        print("===========================================================")
        print("=======================Training data=======================")
        print(train_data)
        print(train_data.shape)
        print("=========================Test data=========================")
        print(train_targets)
        print(train_targets.shape)

    #=======================================================================
    #Normalize the training and test data
    #=======================================================================

    mean = train_data.mean(axis=0)
    train_data -= mean
    std = train_data.std(axis=0)
    train_data /= std

    #=======================================================================
    #Deep learning regression model architecture as a function
    #=======================================================================

    def build_model():
        model = models.Sequential()
        model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
        return model


    #=======================================================================
    #Training and save the validation logs at each fold
    #=======================================================================

    k = 4
    num_val_samples = len(train_data) // k
    num_epochs = 1000
    all_mae_histories = []
    for i in range(k):
        print('processing fold #', i)
    
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    
        partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
        partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
    
        model = build_model()
        history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets), epochs=num_epochs, batch_size=500, verbose=0)
    
        mae_history = history.history['val_mean_absolute_error']
        all_mae_histories.append(mae_history)


    #model = build_model()
    #model.fit(train_data, train_targets,epochs=num_epochs, batch_size=32, verbose=0)
    #model.save('mpc.h5')

    #=======================================================================
    #Calucate and store the mean MAE for each fold
    #=======================================================================

    average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

    #=======================================================================
    #Plot the validation scores using matplotlib
    #=======================================================================

    #plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
    #plt.xlabel('Epochs')
    #plt.ylabel('Validation MAE')
    #plt.show()

    smooth_mae_history = smooth_curve(average_mae_history[10:])
    
    plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')
    plt.show()

def Regression_StateCrimes():
    Train_Data_List = []
    Train_Target_List = []

    with	open('../input/crime_by_state_rt.csv',	'r')	as	f:
        reader	=	csv.DictReader(f,	delimiter=',')
        for	row	in	reader:
            Murder	=	float(row["Murder"])
            Assault_on_women    =   float(row["Assault on women"])
            Kidnapping_and_Abduction  =	float(row["Kidnapping and Abduction"])
            Dacoity	=	float(row["Dacoity"])
            Robbery	=	float(row["Robbery"])
            Arson	=	float(row["Arson"])
            Hurt	=	float(row["Hurt"])
            POA	    =	float(row["Prevention of atrocities (POA) Act"])
            PCR	    =	float(row["Protection of Civil Rights (PCR) Act"])
            Other	=	float(row["Other Crimes Against SCs"])
        
            Train_Data_List.append([Assault_on_women, Kidnapping_and_Abduction, Dacoity, Robbery, Arson, Hurt, POA, PCR, Other])
            Train_Target_List.append(Murder)
    if debug != 0:
        print("=======================Training data=======================")
        print(Train_Data_List)
        print("=========================Test data=========================")
        print(Train_Target_List)


    #=======================================================================
    #Convert the data into 2D Tensors
    #=======================================================================

    train_data =  np.array(Train_Data_List);
    train_targets = np.array(Train_Target_List);

    if debug != 0:
        print("===========================================================")
        print("========================2D tensors=========================")
        print("===========================================================")
        print("=======================Training data=======================")
        print(train_data)
        print(train_data.shape)
        print("=========================Test data=========================")
        print(train_targets)
        print(train_targets.shape)

    #=======================================================================
    #Normalize the training and test data
    #=======================================================================

    mean = train_data.mean(axis=0)
    train_data -= mean
    std = train_data.std(axis=0)
    train_data /= std

    #=======================================================================
    #Deep learning regression model architecture as a function
    #=======================================================================

    def build_model():
        model = models.Sequential()
        model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
        return model


    #=======================================================================
    #Training and save the validation logs at each fold
    #=======================================================================

    k = 4
    num_val_samples = len(train_data) // k
    num_epochs = 200
    all_mae_histories = []
    for i in range(k):
        print('processing fold #', i)
    
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    
        partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
        partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
    
        model = build_model()
        history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets), epochs=num_epochs, batch_size=100, verbose=0)
    
        mae_history = history.history['val_mean_absolute_error']
        all_mae_histories.append(mae_history)


    #model = build_model()
    #model.fit(train_data, train_targets,epochs=num_epochs, batch_size=32, verbose=0)
    #model.save('mpc.h5')

    #=======================================================================
    #Calucate and store the mean MAE for each fold
    #=======================================================================

    average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

    #=======================================================================
    #Plot the validation scores using matplotlib
    #=======================================================================

    #plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
    #plt.xlabel('Epochs')
    #plt.ylabel('Validation MAE')
    #plt.show()

    smooth_mae_history = smooth_curve(average_mae_history[10:])
    
    plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')
    plt.show()

def Prediction_StateCrimes_DLM1():
    Train_Data_List = []
    Train_Target_List = []

    StateList = ["ANDHRA PRADESH", "ARUNACHAL PRADESH", "ASSAM", "BIHAR", "CHHATTISGARH", "GOA", "GUJARAT", "HARYANA", "HIMACHAL PRADESH", "JAMMU & KASHMIR", "JHARKHAND", "KARNATAKA", "KERALA", "MADHYA PRADESH", "MAHARASHTRA", "MANIPUR", "MEGHALAYA", "MIZORAM", "NAGALAND", "ODISHA", "PUNJAB", "RAJASTHAN", "SIKKIM", "TAMIL NADU", "TRIPURA", "UTTAR PRADESH", "UTTARAKHAND", "WEST BENGAL", "A & N ISLANDS", "CHANDIGARH", "D & N HAVELI", "DAMAN & DIU", "DELHI", "LAKSHADWEEP", "PUDUCHERRY"]
    YearList1_train = [2001, 2002, 2003, 2004, 2005]
    YearList2_train = [2002, 2003, 2004, 2005, 2006]
    YearList3_train = [2003, 2004, 2005, 2006, 2007]
    YearList4_train = [2004, 2005, 2006, 2007, 2008]
    YearList5_train = [2005, 2006, 2007, 2008, 2009]
    YearList6_train = [2006, 2007, 2008, 2009, 2010]
    YearList7_train = [2007, 2008, 2009, 2010, 2011]
    YearList_target = [2006, 2007, 2008, 2009, 2010, 2011, 2012]

    DataSet = []

    with	open('../input/crime_by_state_rt.csv',	'r')	as	f:
        reader	=	csv.DictReader(f,	delimiter=',')
        for	row	in	reader:
            State_UT	=	str(row["STATE/UT"])
            Year    =   int(row["Year"])
            Murder	=	float(row["Murder"])
            Assault_on_women    =   float(row["Assault on women"])
            Kidnapping_and_Abduction  =	float(row["Kidnapping and Abduction"])
            Dacoity	=	float(row["Dacoity"])
            Robbery	=	float(row["Robbery"])
            Arson	=	float(row["Arson"])
            Hurt	=	float(row["Hurt"])
            POA	    =	float(row["Prevention of atrocities (POA) Act"])
            PCR	    =	float(row["Protection of Civil Rights (PCR) Act"])
            Other	=	float(row["Other Crimes Against SCs"])

            DataSet.append([State_UT, Year, Murder, Assault_on_women, Kidnapping_and_Abduction, Dacoity, Robbery, Arson, Hurt, POA, PCR, Other])

    # Make the training dataset
    for state in StateList:
        print("==========================Plane 1=============================")
        plane1 = []
        for year in YearList1_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane1.append(item)
                    break
        Train_Data_List.append(plane1)
        print("==========================Plane 2=============================")
        plane2 = []
        for year in YearList2_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane2.append(item)
                    break
        Train_Data_List.append(plane2)
        print("==========================Plane 3=============================")
        plane3 = []
        for year in YearList3_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane3.append(item)
                    break
        Train_Data_List.append(plane3)
        print("==========================Plane 4=============================")
        plane4 = []
        for year in YearList4_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane4.append(item)
                    break
        Train_Data_List.append(plane4)
        print("==========================Plane 5=============================")
        plane5 = []
        for year in YearList5_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane5.append(item)
                    break
        Train_Data_List.append(plane5)
        print("==========================Plane 6=============================")
        plane6 = []
        for year in YearList6_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane6.append(item)
                    break
        Train_Data_List.append(plane6)
        print("==========================Plane 7=============================")
        plane7 = []
        for year in YearList7_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane7.append(item)
                    break
        Train_Data_List.append(plane7)

    # Make the targets dataset
    for state in StateList:
        for record in DataSet:
            if record[0] == state and record[1] == YearList_target[0]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[1]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[2]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[3]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[4]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[5]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[6]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
    #print("======================Training Data======================")
    print(Train_Data_List)
    #print("======================Training Targets======================")  
    print(Train_Target_List)

    #=======================================================================
    #Convert the data into 2D Tensors
    #=======================================================================

    train_data =  np.array(Train_Data_List);
    train_targets = np.array(Train_Target_List);

    if debug != 0:
        print("===========================================================")
        print("========================2D tensors=========================")
        print("===========================================================")
        print("=======================Training data=======================")
        print(train_data)
        print(train_data.shape)
        print("=========================Test data=========================")
        print(train_targets)
        print(train_targets.shape)

    #=======================================================================
    #Normalize the training and test data
    #=======================================================================

    train_data_mean = train_data.mean(axis=0)
    train_data -= train_data_mean
    train_data_std = train_data.std(axis=0)
    train_data /= train_data_std

    train_target_mean = train_targets.mean(axis=0)
    train_targets -= train_target_mean
    train_target_std = train_targets.std(axis=0)
    train_targets /= train_target_std

    #if debug != 0:
    #    print("===========================================================")
    #    print("========================2D tensors=========================")
    #    print("===========================================================")
    #    print("=======================Training data=======================")
    #    print(train_data)
    #    print(train_data.shape)
    #    print("=========================Test data=========================")
    #    print(train_targets)
    #    print(train_targets.shape)

    #=======================================================================
    #Deep learning regression model architecture as a function
    #=======================================================================

    def build_model():
        model = models.Sequential()
        model.add(layers.Dense(512, kernel_regularizer=regularizers.l2(0.005), activation='relu', input_shape=(train_data.shape[1],)))
        model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(0.005), activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
        return model


    #=======================================================================
    #Training and save the validation logs at each fold
    #=======================================================================

    k = 4
    num_val_samples = len(train_data) // k
    num_epochs = 300
    all_mae_histories = []
    for i in range(k):
        print('processing fold #', i)
    
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    
        partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
        partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
    
        model = build_model()
        history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets), epochs=num_epochs, batch_size=10, verbose=0)
    
        mae_history = history.history['val_mean_absolute_error']
        all_mae_histories.append(mae_history)


    #model = build_model()
    #model.fit(train_data, train_targets,epochs=num_epochs, batch_size=32, verbose=0)
    #model.save('mpc.h5')

    #=======================================================================
    #Calucate and store the mean MAE for each fold
    #=======================================================================

    average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

    #=======================================================================
    #Plot the validation scores using matplotlib
    #=======================================================================

    #plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
    #plt.xlabel('Epochs')
    #plt.ylabel('Validation MAE')
    #plt.show()

    smooth_mae_history = smooth_curve(average_mae_history[10:])
    
    plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')
    plt.show()

    #=======================================================================
    # Predict new record using a model and print prediction and actual
    #=======================================================================
    
    val_data = train_data[0 : num_val_samples]
    val_targets = train_targets[0 : num_val_samples]
    
    partial_train_data = np.concatenate([train_data[:0 * num_val_samples], train_data[(0 + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:0 * num_val_samples], train_targets[(0 + 1) * num_val_samples:]], axis=0)
    
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets), epochs=num_epochs, batch_size=10, verbose=0)


    prediction = model.predict(val_data)

    val_targets *= train_target_std
    val_targets += train_target_mean

    prediction *= train_target_std
    prediction += train_target_mean

    AbsoluteError_List = []
    for i in range(num_val_samples):
        print(val_targets[i])
        print(prediction[i])
        print('--------------------------------------')
        AbsoluteError = abs(val_targets[i] - prediction[i])
        AbsoluteError_List.append(AbsoluteError)
        print(AbsoluteError)
        print('=================== '+ str(i) +' ===================')
    SumAbsoluteError = 0
    for error in AbsoluteError_List:
        SumAbsoluteError += error
    MeanAbsoluteError =  SumAbsoluteError / len(AbsoluteError_List)
    print("=================== MeanAbsoluteError: " + str(MeanAbsoluteError))

    plt.plot(range(1, len(val_targets) + 1), val_targets)
    plt.plot(range(1, len(val_targets) + 1), prediction)
    plt.xlabel('Test Samples')
    plt.ylabel('Number of crimes')
    plt.show()

def Prediction_StateCrimes_DLM2():
    Train_Data_List = []
    Train_Target_List = []

    StateList = ["ANDHRA PRADESH", "ARUNACHAL PRADESH", "ASSAM", "BIHAR", "CHHATTISGARH", "GOA", "GUJARAT", "HARYANA", "HIMACHAL PRADESH", "JAMMU & KASHMIR", "JHARKHAND", "KARNATAKA", "KERALA", "MADHYA PRADESH", "MAHARASHTRA", "MANIPUR", "MEGHALAYA", "MIZORAM", "NAGALAND", "ODISHA", "PUNJAB", "RAJASTHAN", "SIKKIM", "TAMIL NADU", "TRIPURA", "UTTAR PRADESH", "UTTARAKHAND", "WEST BENGAL", "A & N ISLANDS", "CHANDIGARH", "D & N HAVELI", "DAMAN & DIU", "DELHI", "LAKSHADWEEP", "PUDUCHERRY"]
    YearList1_train = [2001, 2002, 2003, 2004, 2005]
    YearList2_train = [2002, 2003, 2004, 2005, 2006]
    YearList3_train = [2003, 2004, 2005, 2006, 2007]
    YearList4_train = [2004, 2005, 2006, 2007, 2008]
    YearList5_train = [2005, 2006, 2007, 2008, 2009]
    YearList6_train = [2006, 2007, 2008, 2009, 2010]
    YearList7_train = [2007, 2008, 2009, 2010, 2011]
    YearList_target = [2006, 2007, 2008, 2009, 2010, 2011, 2012]

    DataSet = []

    with	open('../input/crime_by_state_rt.csv',	'r')	as	f:
        reader	=	csv.DictReader(f,	delimiter=',')
        for	row	in	reader:
            State_UT	=	str(row["STATE/UT"])
            Year    =   int(row["Year"])
            Murder	=	float(row["Murder"])
            Assault_on_women    =   float(row["Assault on women"])
            Kidnapping_and_Abduction  =	float(row["Kidnapping and Abduction"])
            Dacoity	=	float(row["Dacoity"])
            Robbery	=	float(row["Robbery"])
            Arson	=	float(row["Arson"])
            Hurt	=	float(row["Hurt"])
            POA	    =	float(row["Prevention of atrocities (POA) Act"])
            PCR	    =	float(row["Protection of Civil Rights (PCR) Act"])
            Other	=	float(row["Other Crimes Against SCs"])

            DataSet.append([State_UT, Year, Murder, Assault_on_women, Kidnapping_and_Abduction, Dacoity, Robbery, Arson, Hurt, POA, PCR, Other])

    # Make the training dataset
    for state in StateList:
        print("==========================Plane 1=============================")
        plane1 = []
        for year in YearList1_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane1.append(item)
                    break
        Train_Data_List.append(plane1)
        print("==========================Plane 2=============================")
        plane2 = []
        for year in YearList2_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane2.append(item)
                    break
        Train_Data_List.append(plane2)
        print("==========================Plane 3=============================")
        plane3 = []
        for year in YearList3_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane3.append(item)
                    break
        Train_Data_List.append(plane3)
        print("==========================Plane 4=============================")
        plane4 = []
        for year in YearList4_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane4.append(item)
                    break
        Train_Data_List.append(plane4)
        print("==========================Plane 5=============================")
        plane5 = []
        for year in YearList5_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane5.append(item)
                    break
        Train_Data_List.append(plane5)
        print("==========================Plane 6=============================")
        plane6 = []
        for year in YearList6_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane6.append(item)
                    break
        Train_Data_List.append(plane6)
        print("==========================Plane 7=============================")
        plane7 = []
        for year in YearList7_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane7.append(item)
                    break
        Train_Data_List.append(plane7)

    # Make the targets dataset
    for state in StateList:
        for record in DataSet:
            if record[0] == state and record[1] == YearList_target[0]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[1]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[2]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[3]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[4]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[5]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[6]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
    #print("======================Training Data======================")
    print(Train_Data_List)
    #print("======================Training Targets======================")  
    print(Train_Target_List)

    #=======================================================================
    #Convert the data into 2D Tensors
    #=======================================================================

    train_data =  np.array(Train_Data_List);
    train_targets = np.array(Train_Target_List);

    if debug != 0:
        print("===========================================================")
        print("========================2D tensors=========================")
        print("===========================================================")
        print("=======================Training data=======================")
        print(train_data)
        print(train_data.shape)
        print("=========================Test data=========================")
        print(train_targets)
        print(train_targets.shape)

    #=======================================================================
    #Normalize the training and test data
    #=======================================================================

    train_data_mean = train_data.mean(axis=0)
    train_data -= train_data_mean
    train_data_std = train_data.std(axis=0)
    train_data /= train_data_std

    train_target_mean = train_targets.mean(axis=0)
    train_targets -= train_target_mean
    train_target_std = train_targets.std(axis=0)
    train_targets /= train_target_std

    #if debug != 0:
    #    print("===========================================================")
    #    print("========================2D tensors=========================")
    #    print("===========================================================")
    #    print("=======================Training data=======================")
    #    print(train_data)
    #    print(train_data.shape)
    #    print("=========================Test data=========================")
    #    print(train_targets)
    #    print(train_targets.shape)

    #=======================================================================
    #Deep learning regression model architecture as a function
    #=======================================================================

    def build_model():
        model = models.Sequential()
        model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(0.01), activation='relu', input_shape=(train_data.shape[1],)))
        model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(0.01), activation='relu'))
        model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(0.01), activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
        return model


    #=======================================================================
    #Training and save the validation logs at each fold
    #=======================================================================

    k = 4
    num_val_samples = len(train_data) // k
    num_epochs = 400
    all_mae_histories = []
    for i in range(k):
        print('processing fold #', i)
    
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    
        partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
        partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
    
        model = build_model()
        history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets), epochs=num_epochs, batch_size=10, verbose=0)
    
        mae_history = history.history['val_mean_absolute_error']
        all_mae_histories.append(mae_history)


    #model = build_model()
    #model.fit(train_data, train_targets,epochs=num_epochs, batch_size=32, verbose=0)
    #model.save('mpc.h5')

    #=======================================================================
    #Calucate and store the mean MAE for each fold
    #=======================================================================

    average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

    #=======================================================================
    #Plot the validation scores using matplotlib
    #=======================================================================

    #plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
    #plt.xlabel('Epochs')
    #plt.ylabel('Validation MAE')
    #plt.show()

    smooth_mae_history = smooth_curve(average_mae_history[10:])
    
    plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')
    plt.show()

    #=======================================================================
    # Predict new record using a model and print prediction and actual
    #=======================================================================
    
    val_data = train_data[0 : num_val_samples]
    val_targets = train_targets[0 : num_val_samples]
    
    partial_train_data = np.concatenate([train_data[:0 * num_val_samples], train_data[(0 + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:0 * num_val_samples], train_targets[(0 + 1) * num_val_samples:]], axis=0)
    
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets), epochs=num_epochs, batch_size=10, verbose=0)


    prediction = model.predict(val_data)

    val_targets *= train_target_std
    val_targets += train_target_mean

    prediction *= train_target_std
    prediction += train_target_mean

    AbsoluteError_List = []
    for i in range(num_val_samples):
        print(val_targets[i])
        print(prediction[i])
        print('--------------------------------------')
        AbsoluteError = abs(val_targets[i] - prediction[i])
        AbsoluteError_List.append(AbsoluteError)
        print(AbsoluteError)
        print('=================== '+ str(i) +' ===================')
    SumAbsoluteError = 0
    for error in AbsoluteError_List:
        SumAbsoluteError += error
    MeanAbsoluteError =  SumAbsoluteError / len(AbsoluteError_List)
    print("=================== MeanAbsoluteError: " + str(MeanAbsoluteError))

    plt.plot(range(1, len(val_targets) + 1), val_targets)
    plt.plot(range(1, len(val_targets) + 1), prediction)
    plt.xlabel('Test Samples')
    plt.ylabel('Number of crimes')
    plt.show()

def Prediction_StateCrimes_DLM3():
    Train_Data_List = []
    Train_Target_List = []

    StateList = ["ANDHRA PRADESH", "ARUNACHAL PRADESH", "ASSAM", "BIHAR", "CHHATTISGARH", "GOA", "GUJARAT", "HARYANA", "HIMACHAL PRADESH", "JAMMU & KASHMIR", "JHARKHAND", "KARNATAKA", "KERALA", "MADHYA PRADESH", "MAHARASHTRA", "MANIPUR", "MEGHALAYA", "MIZORAM", "NAGALAND", "ODISHA", "PUNJAB", "RAJASTHAN", "SIKKIM", "TAMIL NADU", "TRIPURA", "UTTAR PRADESH", "UTTARAKHAND", "WEST BENGAL", "A & N ISLANDS", "CHANDIGARH", "D & N HAVELI", "DAMAN & DIU", "DELHI", "LAKSHADWEEP", "PUDUCHERRY"]
    YearList1_train = [2001, 2002, 2003, 2004, 2005]
    YearList2_train = [2002, 2003, 2004, 2005, 2006]
    YearList3_train = [2003, 2004, 2005, 2006, 2007]
    YearList4_train = [2004, 2005, 2006, 2007, 2008]
    YearList5_train = [2005, 2006, 2007, 2008, 2009]
    YearList6_train = [2006, 2007, 2008, 2009, 2010]
    YearList7_train = [2007, 2008, 2009, 2010, 2011]
    YearList_target = [2006, 2007, 2008, 2009, 2010, 2011, 2012]

    DataSet = []

    with	open('../input/crime_by_state_rt.csv',	'r')	as	f:
        reader	=	csv.DictReader(f,	delimiter=',')
        for	row	in	reader:
            State_UT	=	str(row["STATE/UT"])
            Year    =   int(row["Year"])
            Murder	=	float(row["Murder"])
            Assault_on_women    =   float(row["Assault on women"])
            Kidnapping_and_Abduction  =	float(row["Kidnapping and Abduction"])
            Dacoity	=	float(row["Dacoity"])
            Robbery	=	float(row["Robbery"])
            Arson	=	float(row["Arson"])
            Hurt	=	float(row["Hurt"])
            POA	    =	float(row["Prevention of atrocities (POA) Act"])
            PCR	    =	float(row["Protection of Civil Rights (PCR) Act"])
            Other	=	float(row["Other Crimes Against SCs"])

            DataSet.append([State_UT, Year, Murder, Assault_on_women, Kidnapping_and_Abduction, Dacoity, Robbery, Arson, Hurt, POA, PCR, Other])

    # Make the training dataset
    for state in StateList:
        print("==========================Plane 1=============================")
        plane1 = []
        for year in YearList1_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane1.append(item)
                    break
        Train_Data_List.append(plane1)
        print("==========================Plane 2=============================")
        plane2 = []
        for year in YearList2_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane2.append(item)
                    break
        Train_Data_List.append(plane2)
        print("==========================Plane 3=============================")
        plane3 = []
        for year in YearList3_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane3.append(item)
                    break
        Train_Data_List.append(plane3)
        print("==========================Plane 4=============================")
        plane4 = []
        for year in YearList4_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane4.append(item)
                    break
        Train_Data_List.append(plane4)
        print("==========================Plane 5=============================")
        plane5 = []
        for year in YearList5_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane5.append(item)
                    break
        Train_Data_List.append(plane5)
        print("==========================Plane 6=============================")
        plane6 = []
        for year in YearList6_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane6.append(item)
                    break
        Train_Data_List.append(plane6)
        print("==========================Plane 7=============================")
        plane7 = []
        for year in YearList7_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane7.append(item)
                    break
        Train_Data_List.append(plane7)

    # Make the targets dataset
    for state in StateList:
        for record in DataSet:
            if record[0] == state and record[1] == YearList_target[0]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[1]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[2]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[3]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[4]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[5]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[6]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
    #print("======================Training Data======================")
    print(Train_Data_List)
    #print("======================Training Targets======================")  
    print(Train_Target_List)

    #=======================================================================
    #Convert the data into 2D Tensors
    #=======================================================================

    train_data =  np.array(Train_Data_List);
    train_targets = np.array(Train_Target_List);

    if debug != 0:
        print("===========================================================")
        print("========================2D tensors=========================")
        print("===========================================================")
        print("=======================Training data=======================")
        print(train_data)
        print(train_data.shape)
        print("=========================Test data=========================")
        print(train_targets)
        print(train_targets.shape)

    #=======================================================================
    #Normalize the training and test data
    #=======================================================================

    train_data_mean = train_data.mean(axis=0)
    train_data -= train_data_mean
    train_data_std = train_data.std(axis=0)
    train_data /= train_data_std

    train_target_mean = train_targets.mean(axis=0)
    train_targets -= train_target_mean
    train_target_std = train_targets.std(axis=0)
    train_targets /= train_target_std

    #if debug != 0:
    #    print("===========================================================")
    #    print("========================2D tensors=========================")
    #    print("===========================================================")
    #    print("=======================Training data=======================")
    #    print(train_data)
    #    print(train_data.shape)
    #    print("=========================Test data=========================")
    #    print(train_targets)
    #    print(train_targets.shape)

    #=======================================================================
    #Deep learning regression model architecture as a function
    #=======================================================================

    def build_model():
        model = models.Sequential()
        model.add(layers.Dense(4, kernel_regularizer=regularizers.l2(0.05), activation='relu', input_shape=(train_data.shape[1],)))
        model.add(layers.Dense(4, kernel_regularizer=regularizers.l2(0.05), activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
        return model


    #=======================================================================
    #Training and save the validation logs at each fold
    #=======================================================================

    k = 4
    num_val_samples = len(train_data) // k
    num_epochs = 500
    all_mae_histories = []
    for i in range(k):
        print('processing fold #', i)
    
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    
        partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
        partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
    
        model = build_model()
        history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets), epochs=num_epochs, batch_size=10, verbose=0)
    
        mae_history = history.history['val_mean_absolute_error']
        all_mae_histories.append(mae_history)


    #model = build_model()
    #model.fit(train_data, train_targets,epochs=num_epochs, batch_size=32, verbose=0)
    #model.save('mpc.h5')

    #=======================================================================
    #Calucate and store the mean MAE for each fold
    #=======================================================================

    average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

    #=======================================================================
    #Plot the validation scores using matplotlib
    #=======================================================================

    #plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
    #plt.xlabel('Epochs')
    #plt.ylabel('Validation MAE')
    #plt.show()

    smooth_mae_history = smooth_curve(average_mae_history[10:])
    
    plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')
    plt.show()

    #=======================================================================
    # Predict new record using a model and print prediction and actual
    #=======================================================================
    
    val_data = train_data[0 : num_val_samples]
    val_targets = train_targets[0 : num_val_samples]
    
    partial_train_data = np.concatenate([train_data[:0 * num_val_samples], train_data[(0 + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:0 * num_val_samples], train_targets[(0 + 1) * num_val_samples:]], axis=0)
    
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets), epochs=num_epochs, batch_size=10, verbose=0)


    prediction = model.predict(val_data)

    val_targets *= train_target_std
    val_targets += train_target_mean

    prediction *= train_target_std
    prediction += train_target_mean

    AbsoluteError_List = []
    for i in range(num_val_samples):
        print(val_targets[i])
        print(prediction[i])
        print('--------------------------------------')
        AbsoluteError = abs(val_targets[i] - prediction[i])
        AbsoluteError_List.append(AbsoluteError)
        print(AbsoluteError)
        print('=================== '+ str(i) +' ===================')
    SumAbsoluteError = 0
    for error in AbsoluteError_List:
        SumAbsoluteError += error
    MeanAbsoluteError =  SumAbsoluteError / len(AbsoluteError_List)
    print("=================== MeanAbsoluteError: " + str(MeanAbsoluteError))

    plt.plot(range(1, len(val_targets) + 1), val_targets)
    plt.plot(range(1, len(val_targets) + 1), prediction)
    plt.xlabel('Test Samples')
    plt.ylabel('Number of crimes')
    plt.show()

def Prediction_StateCrimes_SVR_RBF():
    Train_Data_List = []
    Train_Target_List = []

    StateList = ["ANDHRA PRADESH", "ARUNACHAL PRADESH", "ASSAM", "BIHAR", "CHHATTISGARH", "GOA", "GUJARAT", "HARYANA", "HIMACHAL PRADESH", "JAMMU & KASHMIR", "JHARKHAND", "KARNATAKA", "KERALA", "MADHYA PRADESH", "MAHARASHTRA", "MANIPUR", "MEGHALAYA", "MIZORAM", "NAGALAND", "ODISHA", "PUNJAB", "RAJASTHAN", "SIKKIM", "TAMIL NADU", "TRIPURA", "UTTAR PRADESH", "UTTARAKHAND", "WEST BENGAL", "A & N ISLANDS", "CHANDIGARH", "D & N HAVELI", "DAMAN & DIU", "DELHI", "LAKSHADWEEP", "PUDUCHERRY"]
    YearList1_train = [2001, 2002, 2003, 2004, 2005]
    YearList2_train = [2002, 2003, 2004, 2005, 2006]
    YearList3_train = [2003, 2004, 2005, 2006, 2007]
    YearList4_train = [2004, 2005, 2006, 2007, 2008]
    YearList5_train = [2005, 2006, 2007, 2008, 2009]
    YearList6_train = [2006, 2007, 2008, 2009, 2010]
    YearList7_train = [2007, 2008, 2009, 2010, 2011]
    YearList_target = [2006, 2007, 2008, 2009, 2010, 2011, 2012]

    DataSet = []

    with	open('../input/crime_by_state_rt.csv',	'r')	as	f:
        reader	=	csv.DictReader(f,	delimiter=',')
        for	row	in	reader:
            State_UT	=	str(row["STATE/UT"])
            Year    =   int(row["Year"])
            Murder	=	float(row["Murder"])
            Assault_on_women    =   float(row["Assault on women"])
            Kidnapping_and_Abduction  =	float(row["Kidnapping and Abduction"])
            Dacoity	=	float(row["Dacoity"])
            Robbery	=	float(row["Robbery"])
            Arson	=	float(row["Arson"])
            Hurt	=	float(row["Hurt"])
            POA	    =	float(row["Prevention of atrocities (POA) Act"])
            PCR	    =	float(row["Protection of Civil Rights (PCR) Act"])
            Other	=	float(row["Other Crimes Against SCs"])

            DataSet.append([State_UT, Year, Murder, Assault_on_women, Kidnapping_and_Abduction, Dacoity, Robbery, Arson, Hurt, POA, PCR, Other])

    # Make the training dataset
    for state in StateList:
        print("==========================Plane 1=============================")
        plane1 = []
        for year in YearList1_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane1.append(item)
                    break
        Train_Data_List.append(plane1)
        print("==========================Plane 2=============================")
        plane2 = []
        for year in YearList2_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane2.append(item)
                    break
        Train_Data_List.append(plane2)
        print("==========================Plane 3=============================")
        plane3 = []
        for year in YearList3_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane3.append(item)
                    break
        Train_Data_List.append(plane3)
        print("==========================Plane 4=============================")
        plane4 = []
        for year in YearList4_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane4.append(item)
                    break
        Train_Data_List.append(plane4)
        print("==========================Plane 5=============================")
        plane5 = []
        for year in YearList5_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane5.append(item)
                    break
        Train_Data_List.append(plane5)
        print("==========================Plane 6=============================")
        plane6 = []
        for year in YearList6_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane6.append(item)
                    break
        Train_Data_List.append(plane6)
        print("==========================Plane 7=============================")
        plane7 = []
        for year in YearList7_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane7.append(item)
                    break
        Train_Data_List.append(plane7)

    # Make the targets dataset
    for state in StateList:
        for record in DataSet:
            if record[0] == state and record[1] == YearList_target[0]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[1]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[2]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[3]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[4]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[5]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[6]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
    #print("======================Training Data======================")
    print(Train_Data_List)
    #print("======================Training Targets======================")  
    print(Train_Target_List)

    #=======================================================================
    #Convert the data into 2D Tensors
    #=======================================================================

    train_data =  np.array(Train_Data_List);
    train_targets = np.array(Train_Target_List);

    if debug != 0:
        print("===========================================================")
        print("========================2D tensors=========================")
        print("===========================================================")
        print("=======================Training data=======================")
        print(train_data)
        print(train_data.shape)
        print("=========================Test data=========================")
        print(train_targets)
        print(train_targets.shape)

    #=======================================================================
    #Normalize the training and test data
    #=======================================================================

    train_data_mean = train_data.mean(axis=0)
    train_data -= train_data_mean
    train_data_std = train_data.std(axis=0)
    train_data /= train_data_std

    train_target_mean = train_targets.mean(axis=0)
    train_targets -= train_target_mean
    train_target_std = train_targets.std(axis=0)
    train_targets /= train_target_std

    #if debug != 0:
    #    print("===========================================================")
    #    print("========================2D tensors=========================")
    #    print("===========================================================")
    #    print("=======================Training data=======================")
    #    print(train_data)
    #    print(train_data.shape)
    #    print("=========================Test data=========================")
    #    print(train_targets)
    #    print(train_targets.shape)

    #=======================================================================
    # Support vector regression model as a function
    #=======================================================================

    def build_model():
        model_svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        return model_svr_rbf

    k = 4
    num_val_samples = len(train_data) // k

    #=======================================================================
    # Predict new record using a model and print prediction and actual
    #=======================================================================
    
    val_data = train_data[0 : num_val_samples]
    val_targets = train_targets[0 : num_val_samples]
    
    partial_train_data = np.concatenate([train_data[:0 * num_val_samples], train_data[(0 + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:0 * num_val_samples], train_targets[(0 + 1) * num_val_samples:]], axis=0)
    
    model = build_model()
    prediction = model.fit(partial_train_data, partial_train_targets).predict(val_data)

    val_targets *= train_target_std
    val_targets += train_target_mean

    prediction *= train_target_std
    prediction += train_target_mean

    AbsoluteError_List = []
    for i in range(num_val_samples):
        print(val_targets[i])
        print(prediction[i])
        print('--------------------------------------')
        AbsoluteError = abs(val_targets[i] - prediction[i])
        AbsoluteError_List.append(AbsoluteError)
        print(AbsoluteError)
        print('=================== '+ str(i) +' ===================')
    SumAbsoluteError = 0
    for error in AbsoluteError_List:
        SumAbsoluteError += error
    MeanAbsoluteError =  SumAbsoluteError / len(AbsoluteError_List)
    print("=================== MeanAbsoluteError: " + str(MeanAbsoluteError))

    plt.plot(range(1, len(val_targets) + 1), val_targets)
    plt.plot(range(1, len(val_targets) + 1), prediction)
    plt.xlabel('Test Samples')
    plt.ylabel('Number of crimes')
    plt.show()

def Prediction_StateCrimes_SVR_Linear():
    Train_Data_List = []
    Train_Target_List = []

    StateList = ["ANDHRA PRADESH", "ARUNACHAL PRADESH", "ASSAM", "BIHAR", "CHHATTISGARH", "GOA", "GUJARAT", "HARYANA", "HIMACHAL PRADESH", "JAMMU & KASHMIR", "JHARKHAND", "KARNATAKA", "KERALA", "MADHYA PRADESH", "MAHARASHTRA", "MANIPUR", "MEGHALAYA", "MIZORAM", "NAGALAND", "ODISHA", "PUNJAB", "RAJASTHAN", "SIKKIM", "TAMIL NADU", "TRIPURA", "UTTAR PRADESH", "UTTARAKHAND", "WEST BENGAL", "A & N ISLANDS", "CHANDIGARH", "D & N HAVELI", "DAMAN & DIU", "DELHI", "LAKSHADWEEP", "PUDUCHERRY"]
    YearList1_train = [2001, 2002, 2003, 2004, 2005]
    YearList2_train = [2002, 2003, 2004, 2005, 2006]
    YearList3_train = [2003, 2004, 2005, 2006, 2007]
    YearList4_train = [2004, 2005, 2006, 2007, 2008]
    YearList5_train = [2005, 2006, 2007, 2008, 2009]
    YearList6_train = [2006, 2007, 2008, 2009, 2010]
    YearList7_train = [2007, 2008, 2009, 2010, 2011]
    YearList_target = [2006, 2007, 2008, 2009, 2010, 2011, 2012]

    DataSet = []

    with	open('../input/crime_by_state_rt.csv',	'r')	as	f:
        reader	=	csv.DictReader(f,	delimiter=',')
        for	row	in	reader:
            State_UT	=	str(row["STATE/UT"])
            Year    =   int(row["Year"])
            Murder	=	float(row["Murder"])
            Assault_on_women    =   float(row["Assault on women"])
            Kidnapping_and_Abduction  =	float(row["Kidnapping and Abduction"])
            Dacoity	=	float(row["Dacoity"])
            Robbery	=	float(row["Robbery"])
            Arson	=	float(row["Arson"])
            Hurt	=	float(row["Hurt"])
            POA	    =	float(row["Prevention of atrocities (POA) Act"])
            PCR	    =	float(row["Protection of Civil Rights (PCR) Act"])
            Other	=	float(row["Other Crimes Against SCs"])

            DataSet.append([State_UT, Year, Murder, Assault_on_women, Kidnapping_and_Abduction, Dacoity, Robbery, Arson, Hurt, POA, PCR, Other])

    # Make the training dataset
    for state in StateList:
        print("==========================Plane 1=============================")
        plane1 = []
        for year in YearList1_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane1.append(item)
                    break
        Train_Data_List.append(plane1)
        print("==========================Plane 2=============================")
        plane2 = []
        for year in YearList2_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane2.append(item)
                    break
        Train_Data_List.append(plane2)
        print("==========================Plane 3=============================")
        plane3 = []
        for year in YearList3_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane3.append(item)
                    break
        Train_Data_List.append(plane3)
        print("==========================Plane 4=============================")
        plane4 = []
        for year in YearList4_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane4.append(item)
                    break
        Train_Data_List.append(plane4)
        print("==========================Plane 5=============================")
        plane5 = []
        for year in YearList5_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane5.append(item)
                    break
        Train_Data_List.append(plane5)
        print("==========================Plane 6=============================")
        plane6 = []
        for year in YearList6_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane6.append(item)
                    break
        Train_Data_List.append(plane6)
        print("==========================Plane 7=============================")
        plane7 = []
        for year in YearList7_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane7.append(item)
                    break
        Train_Data_List.append(plane7)

    # Make the targets dataset
    for state in StateList:
        for record in DataSet:
            if record[0] == state and record[1] == YearList_target[0]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[1]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[2]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[3]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[4]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[5]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[6]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
    #print("======================Training Data======================")
    print(Train_Data_List)
    #print("======================Training Targets======================")  
    print(Train_Target_List)

    #=======================================================================
    #Convert the data into 2D Tensors
    #=======================================================================

    train_data =  np.array(Train_Data_List);
    train_targets = np.array(Train_Target_List);

    if debug != 0:
        print("===========================================================")
        print("========================2D tensors=========================")
        print("===========================================================")
        print("=======================Training data=======================")
        print(train_data)
        print(train_data.shape)
        print("=========================Test data=========================")
        print(train_targets)
        print(train_targets.shape)

    #=======================================================================
    #Normalize the training and test data
    #=======================================================================

    train_data_mean = train_data.mean(axis=0)
    train_data -= train_data_mean
    train_data_std = train_data.std(axis=0)
    train_data /= train_data_std

    train_target_mean = train_targets.mean(axis=0)
    train_targets -= train_target_mean
    train_target_std = train_targets.std(axis=0)
    train_targets /= train_target_std

    #if debug != 0:
    #    print("===========================================================")
    #    print("========================2D tensors=========================")
    #    print("===========================================================")
    #    print("=======================Training data=======================")
    #    print(train_data)
    #    print(train_data.shape)
    #    print("=========================Test data=========================")
    #    print(train_targets)
    #    print(train_targets.shape)

    #=======================================================================
    # Support vector regression model as a function
    #=======================================================================

    def build_model():
        model_svr_lin = SVR(kernel='linear', C=100, gamma='auto')
        return model_svr_lin

    k = 4
    num_val_samples = len(train_data) // k

    #=======================================================================
    # Predict new record using a model and print prediction and actual
    #=======================================================================
    
    val_data = train_data[0 : num_val_samples]
    val_targets = train_targets[0 : num_val_samples]
    
    partial_train_data = np.concatenate([train_data[:0 * num_val_samples], train_data[(0 + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:0 * num_val_samples], train_targets[(0 + 1) * num_val_samples:]], axis=0)
    
    model = build_model()
    prediction = model.fit(partial_train_data, partial_train_targets).predict(val_data)

    val_targets *= train_target_std
    val_targets += train_target_mean

    prediction *= train_target_std
    prediction += train_target_mean

    AbsoluteError_List = []
    for i in range(num_val_samples):
        print(val_targets[i])
        print(prediction[i])
        print('--------------------------------------')
        AbsoluteError = abs(val_targets[i] - prediction[i])
        AbsoluteError_List.append(AbsoluteError)
        print(AbsoluteError)
        print('=================== '+ str(i) +' ===================')
    SumAbsoluteError = 0
    for error in AbsoluteError_List:
        SumAbsoluteError += error
    MeanAbsoluteError =  SumAbsoluteError / len(AbsoluteError_List)
    print("=================== MeanAbsoluteError: " + str(MeanAbsoluteError))

    plt.plot(range(1, len(val_targets) + 1), val_targets)
    plt.plot(range(1, len(val_targets) + 1), prediction)
    plt.xlabel('Test Samples')
    plt.ylabel('Number of crimes')
    plt.show()

def Prediction_StateCrimes_SVR_Polynomial():
    Train_Data_List = []
    Train_Target_List = []

    StateList = ["ANDHRA PRADESH", "ARUNACHAL PRADESH", "ASSAM", "BIHAR", "CHHATTISGARH", "GOA", "GUJARAT", "HARYANA", "HIMACHAL PRADESH", "JAMMU & KASHMIR", "JHARKHAND", "KARNATAKA", "KERALA", "MADHYA PRADESH", "MAHARASHTRA", "MANIPUR", "MEGHALAYA", "MIZORAM", "NAGALAND", "ODISHA", "PUNJAB", "RAJASTHAN", "SIKKIM", "TAMIL NADU", "TRIPURA", "UTTAR PRADESH", "UTTARAKHAND", "WEST BENGAL", "A & N ISLANDS", "CHANDIGARH", "D & N HAVELI", "DAMAN & DIU", "DELHI", "LAKSHADWEEP", "PUDUCHERRY"]
    YearList1_train = [2001, 2002, 2003, 2004, 2005]
    YearList2_train = [2002, 2003, 2004, 2005, 2006]
    YearList3_train = [2003, 2004, 2005, 2006, 2007]
    YearList4_train = [2004, 2005, 2006, 2007, 2008]
    YearList5_train = [2005, 2006, 2007, 2008, 2009]
    YearList6_train = [2006, 2007, 2008, 2009, 2010]
    YearList7_train = [2007, 2008, 2009, 2010, 2011]
    YearList_target = [2006, 2007, 2008, 2009, 2010, 2011, 2012]

    DataSet = []

    with	open('../input/crime_by_state_rt.csv',	'r')	as	f:
        reader	=	csv.DictReader(f,	delimiter=',')
        for	row	in	reader:
            State_UT	=	str(row["STATE/UT"])
            Year    =   int(row["Year"])
            Murder	=	float(row["Murder"])
            Assault_on_women    =   float(row["Assault on women"])
            Kidnapping_and_Abduction  =	float(row["Kidnapping and Abduction"])
            Dacoity	=	float(row["Dacoity"])
            Robbery	=	float(row["Robbery"])
            Arson	=	float(row["Arson"])
            Hurt	=	float(row["Hurt"])
            POA	    =	float(row["Prevention of atrocities (POA) Act"])
            PCR	    =	float(row["Protection of Civil Rights (PCR) Act"])
            Other	=	float(row["Other Crimes Against SCs"])

            DataSet.append([State_UT, Year, Murder, Assault_on_women, Kidnapping_and_Abduction, Dacoity, Robbery, Arson, Hurt, POA, PCR, Other])

    # Make the training dataset
    for state in StateList:
        print("==========================Plane 1=============================")
        plane1 = []
        for year in YearList1_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane1.append(item)
                    break
        Train_Data_List.append(plane1)
        print("==========================Plane 2=============================")
        plane2 = []
        for year in YearList2_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane2.append(item)
                    break
        Train_Data_List.append(plane2)
        print("==========================Plane 3=============================")
        plane3 = []
        for year in YearList3_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane3.append(item)
                    break
        Train_Data_List.append(plane3)
        print("==========================Plane 4=============================")
        plane4 = []
        for year in YearList4_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane4.append(item)
                    break
        Train_Data_List.append(plane4)
        print("==========================Plane 5=============================")
        plane5 = []
        for year in YearList5_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane5.append(item)
                    break
        Train_Data_List.append(plane5)
        print("==========================Plane 6=============================")
        plane6 = []
        for year in YearList6_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane6.append(item)
                    break
        Train_Data_List.append(plane6)
        print("==========================Plane 7=============================")
        plane7 = []
        for year in YearList7_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane7.append(item)
                    break
        Train_Data_List.append(plane7)

    # Make the targets dataset
    for state in StateList:
        for record in DataSet:
            if record[0] == state and record[1] == YearList_target[0]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[1]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[2]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[3]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[4]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[5]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[6]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
    #print("======================Training Data======================")
    print(Train_Data_List)
    #print("======================Training Targets======================")  
    print(Train_Target_List)

    #=======================================================================
    #Convert the data into 2D Tensors
    #=======================================================================

    train_data =  np.array(Train_Data_List);
    train_targets = np.array(Train_Target_List);

    if debug != 0:
        print("===========================================================")
        print("========================2D tensors=========================")
        print("===========================================================")
        print("=======================Training data=======================")
        print(train_data)
        print(train_data.shape)
        print("=========================Test data=========================")
        print(train_targets)
        print(train_targets.shape)

    #=======================================================================
    #Normalize the training and test data
    #=======================================================================

    train_data_mean = train_data.mean(axis=0)
    train_data -= train_data_mean
    train_data_std = train_data.std(axis=0)
    train_data /= train_data_std

    train_target_mean = train_targets.mean(axis=0)
    train_targets -= train_target_mean
    train_target_std = train_targets.std(axis=0)
    train_targets /= train_target_std

    #if debug != 0:
    #    print("===========================================================")
    #    print("========================2D tensors=========================")
    #    print("===========================================================")
    #    print("=======================Training data=======================")
    #    print(train_data)
    #    print(train_data.shape)
    #    print("=========================Test data=========================")
    #    print(train_targets)
    #    print(train_targets.shape)

    #=======================================================================
    # Support vector regression model as a function
    #=======================================================================

    def build_model():
        model_svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1)
        return model_svr_poly

    k = 4
    num_val_samples = len(train_data) // k

    #=======================================================================
    # Predict new record using a model and print prediction and actual
    #=======================================================================
    
    val_data = train_data[0 : num_val_samples]
    val_targets = train_targets[0 : num_val_samples]
    
    partial_train_data = np.concatenate([train_data[:0 * num_val_samples], train_data[(0 + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:0 * num_val_samples], train_targets[(0 + 1) * num_val_samples:]], axis=0)
    
    model = build_model()
    prediction = model.fit(partial_train_data, partial_train_targets).predict(val_data)

    val_targets *= train_target_std
    val_targets += train_target_mean

    prediction *= train_target_std
    prediction += train_target_mean

    AbsoluteError_List = []
    for i in range(num_val_samples):
        print(val_targets[i])
        print(prediction[i])
        print('--------------------------------------')
        AbsoluteError = abs(val_targets[i] - prediction[i])
        AbsoluteError_List.append(AbsoluteError)
        print(AbsoluteError)
        print('=================== '+ str(i) +' ===================')
    SumAbsoluteError = 0
    for error in AbsoluteError_List:
        SumAbsoluteError += error
    MeanAbsoluteError =  SumAbsoluteError / len(AbsoluteError_List)
    print("=================== MeanAbsoluteError: " + str(MeanAbsoluteError))

    plt.plot(range(1, len(val_targets) + 1), val_targets)
    plt.plot(range(1, len(val_targets) + 1), prediction)
    plt.xlabel('Test Samples')
    plt.ylabel('Number of crimes')
    plt.show()

def Prediction_StateCrimes_k_fold():
    Train_Data_List = []
    Train_Target_List = []

    StateList = ["ANDHRA PRADESH", "ARUNACHAL PRADESH", "ASSAM", "BIHAR", "CHHATTISGARH", "GOA", "GUJARAT", "HARYANA", "HIMACHAL PRADESH", "JAMMU & KASHMIR", "JHARKHAND", "KARNATAKA", "KERALA", "MADHYA PRADESH", "MAHARASHTRA", "MANIPUR", "MEGHALAYA", "MIZORAM", "NAGALAND", "ODISHA", "PUNJAB", "RAJASTHAN", "SIKKIM", "TAMIL NADU", "TRIPURA", "UTTAR PRADESH", "UTTARAKHAND", "WEST BENGAL", "A & N ISLANDS", "CHANDIGARH", "D & N HAVELI", "DAMAN & DIU", "DELHI", "LAKSHADWEEP", "PUDUCHERRY"]
    YearList1_train = [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008]
    YearList2_train = [2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009]
    YearList3_train = [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010]
    YearList4_train = [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011]
    YearList_target = [2009, 2010, 2011, 2012]

    DataSet = []

    with	open('../input/crime_by_state_rt.csv',	'r')	as	f:
        reader	=	csv.DictReader(f,	delimiter=',')
        for	row	in	reader:
            State_UT	=	str(row["STATE/UT"])
            Year    =   int(row["Year"])
            Murder	=	float(row["Murder"])
            Assault_on_women    =   float(row["Assault on women"])
            Kidnapping_and_Abduction  =	float(row["Kidnapping and Abduction"])
            Dacoity	=	float(row["Dacoity"])
            Robbery	=	float(row["Robbery"])
            Arson	=	float(row["Arson"])
            Hurt	=	float(row["Hurt"])
            POA	    =	float(row["Prevention of atrocities (POA) Act"])
            PCR	    =	float(row["Protection of Civil Rights (PCR) Act"])
            Other	=	float(row["Other Crimes Against SCs"])

            DataSet.append([State_UT, Year, Murder, Assault_on_women, Kidnapping_and_Abduction, Dacoity, Robbery, Arson, Hurt, POA, PCR, Other])

    # Make the training dataset
    for state in StateList:
        print("==========================Plane 1=============================")
        plane1 = []
        for year in YearList1_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane1.append(item)
                    break
        Train_Data_List.append(plane1)
        print("==========================Plane 2=============================")
        plane2 = []
        for year in YearList2_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane2.append(item)
                    break
        Train_Data_List.append(plane2)
        print("==========================Plane 3=============================")
        plane3 = []
        for year in YearList3_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane3.append(item)
                    break
        Train_Data_List.append(plane3)
        print("==========================Plane 4=============================")
        plane4 = []
        for year in YearList4_train:
            for record in DataSet:
                if record[0] == state and record[1] == year:
                    print(record[2 : ])
                    for item in record[2 : ]:
                        plane4.append(item)
                    break
        Train_Data_List.append(plane4)

    # Make the targets dataset
    for state in StateList:
        for record in DataSet:
            if record[0] == state and record[1] == YearList_target[0]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[1]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[2]:
                Train_Target_List.append(record[2])  # Crimes start from index 2
            if record[0] == state and record[1] == YearList_target[3]:
                Train_Target_List.append(record[2])  # Crimes start from index 2

    #print("======================Training Data======================")
    print(Train_Data_List)
    #print("======================Training Targets======================")
    print(Train_Target_List)

    #=======================================================================
    #Convert the data into 2D Tensors
    #=======================================================================

    train_data =  np.array(Train_Data_List);
    train_targets = np.array(Train_Target_List);

    if debug != 0:
        print("===========================================================")
        print("========================2D tensors=========================")
        print("===========================================================")
        print("=======================Training data=======================")
        print(train_data)
        print(train_data.shape)
        print("=========================Test data=========================")
        print(train_targets)
        print(train_targets.shape)

    #=======================================================================
    #Normalize the training and test data
    #=======================================================================

    train_data_mean = train_data.mean(axis=0)
    train_data -= train_data_mean
    train_data_std = train_data.std(axis=0)
    train_data /= train_data_std

    train_target_mean = train_targets.mean(axis=0)
    train_targets -= train_target_mean
    train_target_std = train_targets.std(axis=0)
    train_targets /= train_target_std

    #if debug != 0:
    #    print("===========================================================")
    #    print("========================2D tensors=========================")
    #    print("===========================================================")
    #    print("=======================Training data=======================")
    #    print(train_data)
    #    print(train_data.shape)
    #    print("=========================Test data=========================")
    #    print(train_targets)
    #    print(train_targets.shape)

    #=======================================================================
    #Deep learning regression model architecture as a function
    #=======================================================================

    def build_model():
        model = models.Sequential()
        model.add(layers.Dense(512, activation='relu', input_shape=(train_data.shape[1],)))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
        return model


    #=======================================================================
    #Training and save the validation logs at each fold
    #=======================================================================
    
    # These three are used in the next k-fold validation as well
    k = 4
    num_val_samples = len(train_data) // k
    num_epochs = 400

    all_mae_histories = []
    for i in range(k):
        print('processing fold #', i)
    
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    
        partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
        partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
    
        model = build_model()
        history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets), epochs=num_epochs, batch_size=10, verbose=0)
    
        mae_history = history.history['val_mean_absolute_error']
        all_mae_histories.append(mae_history)


    #model = build_model()
    #model.fit(train_data, train_targets,epochs=num_epochs, batch_size=32, verbose=0)
    #model.save('mpc.h5')

    #=======================================================================
    #Calucate and store the mean MAE for each fold
    #=======================================================================

    average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

    #=======================================================================
    #Plot the validation scores using matplotlib
    #=======================================================================

    #plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
    #plt.xlabel('Epochs')
    #plt.ylabel('Validation MAE')
    #plt.show()

    smooth_mae_history = smooth_curve(average_mae_history[10:])
    
    plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')
    plt.show()

    #=======================================================================
    # Predict new record using a model and print prediction and actual
    #=======================================================================
    
    all_val_targets = []
    all_predictions = []

    for i in range(k):
        print('processing fold #', i)
    
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

        partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
        partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
    
        model = build_model()
        history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets), epochs=num_epochs, batch_size=10, verbose=0)
        
        prediction = model.predict(val_data)

        val_targets *= train_target_std
        val_targets += train_target_mean

        prediction *= train_target_std
        prediction += train_target_mean

        for item in val_targets:
            all_val_targets.append(item)
        print("======================all_val_targets===========================")
        print(all_val_targets)

        for item in prediction:
            if float(item[0]) > 4000:
                all_predictions.append(0)
            else:
                all_predictions.append(float(item[0]))
        print("======================all_predictions===========================")
        print(all_predictions)

    AbsoluteError_List = []
    for i in range(len(all_val_targets)):
        print(all_val_targets[i])
        print(all_predictions[i])
        print('--------------------------------------')
        AbsoluteError = abs(all_val_targets[i] - all_predictions[i])
        AbsoluteError_List.append(AbsoluteError)
        print(AbsoluteError)
        print('=================== '+ str(i) +' ===================')
    SumAbsoluteError = 0
    for error in AbsoluteError_List:
        SumAbsoluteError += error
    MeanAbsoluteError =  SumAbsoluteError / len(AbsoluteError_List)
    print("=================== MeanAbsoluteError: " + str(MeanAbsoluteError))

    plt.plot(range(1, len(all_val_targets) + 1), all_val_targets)
    plt.plot(range(1, len(all_val_targets) + 1), all_predictions)
    plt.xlabel('Test Samples')
    plt.ylabel('Number of crimes')
    plt.show()


def main(): 
    print("0. Exit")
    print("1. create Regression model for district level crimes")
    print("2. create Regression model for state level crimes")
    print("3. Crime prediction - DeepLearning Model DLM-1")
    print("4. Crime prediction - DeepLearning Model DLM-2")
    print("5. Crime prediction - DeepLearning Model DLM-3")
    print("6. Crime prediction - Support Vector Regression - RBF Kernel")
    print("7. Crime prediction - Support Vector Regression - Linear Kernel")
    print("8. Crime prediction - Support Vector Regression - Polynomial Kernel")
    print("...")
    choice = eval(input("Do which one? "))

    while(choice != 0):
        if choice == 1:
            Regression_DistrictCrimes()
            print("Done...")
        if choice == 2:
            Regression_StateCrimes()
            print("Done...")
        if choice == 3:
            Prediction_StateCrimes_DLM1()
            print("Done...")
        if choice == 4:
            Prediction_StateCrimes_DLM2()
            print("Done...")
        if choice == 5:
            Prediction_StateCrimes_DLM3()
            print("Done...")
        if choice == 6:
            Prediction_StateCrimes_SVR_RBF()
            print("Done...")
        if choice == 7:
            Prediction_StateCrimes_SVR_Linear()
            print("Done...")
        if choice == 8:
            Prediction_StateCrimes_SVR_Polynomial()
            print("Done...")
        print("0. Exit")
        print("1. create Regression model for district level crimes")
        print("2. create Regression model for state level crimes")
        print("3. Crime prediction - DeepLearning Model DLM-1")
        print("4. Crime prediction - DeepLearning Model DLM-2")
        print("5. Crime prediction - DeepLearning Model DLM-3")
        print("6. Crime prediction - Support Vector Regression - RBF Kernel")
        print("7. Crime prediction - Support Vector Regression - Linear Kernel")
        print("8. Crime prediction - Support Vector Regression - Polynomial Kernel")
        print("...")
        choice = eval(input("Do which one? "))

main()