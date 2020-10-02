import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import math
import joblib
import os.path
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#defining minmax scaler
sc = MinMaxScaler(feature_range = (0, 1))

#Define STATIC Variables
TRAIN_CSV = "/kaggle/input/predict-traffic-of-lte-network/train.csv"
TEST_CSV = "/kaggle/input/predict-traffic-of-lte-network/test.csv"
EPOCH = 200
BATCHSIZE = 50
MODEL_PKL = "/kaggle/working/model.pkl"
MODEL_INPUT_PKL = "/kaggle/input/final-data/model.pk"
VALID_PLOT = "/kaggle/working/validation.png"
PRED_PLOT = "/kaggle/working/predict.png"
N_PREDICT = 200


def preprocessing(train, test, cell="Cell_000113"):
    
    #reading train, test datasets
    df = pd.read_csv(train)
    df_test = pd.read_csv(test)
    
    #filter the dataset to provided cell
    df = df[df["CellName"] == cell]
    df_test = df_test[df_test["CellName"] == cell]
    
    #combining both test and train for adding datatime and droping [date, hour] columns
    total = [df, df_test]
    dataset = pd.concat(total)
    dataset["Date"] = pd.to_datetime(dataset.Date.astype(str))
    dataset["Hour"] = pd.to_timedelta(dataset.Hour, unit="h")
    dataset["DateTime"] = pd.to_datetime(dataset.Date + dataset.Hour)
    dataset = dataset.drop(["Hour", "Date", "CellName"], axis=1)

    #Making DateTime column as index and soring them accordingly
    dataset = dataset.set_index("DateTime")
    #dataset = dataset.sort_index()
    
    #spliting data to train,test and predict
    train, test = dataset[:len(df)], dataset[len(df):]
    
    return train,test
    
def model(X_train, y_train):
    
    #Model Creation
    regressor = Sequential()
    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units = 1))
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    regressor.fit(X_train, y_train, epochs = EPOCH, batch_size = BATCHSIZE)
    
    return regressor

def train_model(model, train):
    
    train = train.sort_index()
    
    ##fitting with minmaxscaler
    train = sc.fit_transform(train)
    
    X_train = []
    y_train = []
    for i in range(24, train.size):
        X_train.append(train[i-24:i, 0])
        y_train.append(train[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    #reshaping data
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    #model training
    model = model(X_train, y_train)
    
    #writing model to pickel file
    if MODEL_PKL != "":
        store_model(model)
        
    return model
    
def store_model(model):
    
    #dumping model to pkl file
    joblib.dump(model, MODEL_PKL) 
    
def model_validation(model, train, test):
    
    test = test.sort_index()
    
    data = pd.concat([train,test])
    
    data_test = data[len(data) - len(test) - 24:]
        
    #fitting with minmaxscaler
    data_test = sc.fit_transform(data_test)
        
    X_test = []
    for i in range(24, len(data_test)):
        X_test.append(data_test[i-24:i,:])
    X_test = np.array(X_test)
    
    predicts = regressor.predict(X_test)
    predicts = sc.inverse_transform(predicts)
    
    plot_validation(test.values, predicts)
    
    return X_test[-1]
    
def plot_validation(original, predict):
    
    plt.plot(original, color = 'red', label = 'Real Traffic ')
    plt.plot(predict, color = 'blue', label = 'Predicted Traffic')
    plt.title('Model validation')
    plt.xlabel('Time')
    plt.ylabel('Traffic')
    plt.legend()
    plt.show()
    
    if VALID_PLOT != "":
        plt.savefig(VALID_PLOT, format='png')

def sliding_window(model, input_sample, n_predict):
    
    list_pred = []
    for i in range(n_predict):
        predict = model.predict(input_sample)
        a = sc.inverse_transform(predict)
        list_pred.append(a[0][0].tolist())
        predict = np.reshape(predict, (1, predict.shape[0], 1))
        input_sample = input_sample[:,1:,:]
        input_sample = np.concatenate((input_sample,predict),axis=1)

    return list_pred

def prediction(pred, model=None, pkl=None):
    
    if pkl:
        model = joblib.load(pkl)
    
    #fitting with minmaxscaler
    #pred = sc.fit_transform(pred)
    
    pred_reshaped = np.reshape(pred, (1, pred.shape[0], 1))
        
    list_pred = sliding_window(model=model, input_sample=pred_reshaped, n_predict=N_PREDICT)
    
    return list_pred
    

def AMF_cal(pred_list):
    
    amf = list(map(lambda x : math.ceil((x*0.2)/100), pred_list))
    
    amf_list = []
    for i in range(0,len(amf),10):
        #amf_list.extend([max(amf[i:i+10]) for i in range(10)])
        amf_list.extend([max(amf[i:i+10]) for _ in range(10)])
    #print(amf_list)
    
    pre = list(map(lambda x: x*0.2, pred_list))
    
    pred_plot(amf_list, pre)
    return amf_list

def pred_plot(amf, predlist):
    
    x = [i for i in range(0,len(amf),10)]
    fig,ax1 = plt.subplots()
    
    ax2 = ax1.twinx()
    plot1, = ax2.plot(amf, color='red', label='Required AMF')
    plot2, = ax1.plot(predlist, color='blue', label='Predicted Traffic')
    
    plot = [plot1, plot2]
    
    #plt.figure(figsize=(15, 7))
    plt.title('Prediction')
    plt.xlabel('Time')
    ax1.set_xticks(np.arange(0, len(amf), 10))
    #plt.xticks(rotation=180)
    ax1.set_xlabel('Time(s)')
    ax2.set_yticks(np.arange(0, max(amf), 1))
    ax2.set_ylabel('AMF', color='red')
    ax1.set_ylabel('Traffic', color='blue')
    plt.legend(plot,[plots.get_label() for plots in plot])
    plt.show()
    
    if PRED_PLOT != "":
        plt.savefig(PRED_PLOT, format='png')

if __name__ == "__main__":
    
    #Data pre-processing
    train, test = preprocessing(train=TRAIN_CSV, test=TEST_CSV)
    print(f'train shape: {train.shape}, test shape: {test.shape}')
    
    
    if os.path.exists(MODEL_INPUT_PKL):
        #for loading model 
        regressor = joblib.load(MODEL_INPUT_PKL)
    else:
        #Model training
        regressor = train_model(model, train)
        
    #model validation
    pred = model_validation(model=regressor, train=train, test=test)
    
    #Prediction
    list_pred = prediction(pred=pred, model=regressor)
    print(f'list_pred: {list_pred}')
        
    #calculating Amf
    amf = AMF_cal(list_pred)
    print(f'amf: {amf}')
    
    
    