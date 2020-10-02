#importing Libraries
import re
import string
import pandas as pd
from pickle import dump
from pickle import load
from numpy import array
from datetime import datetime, timedelta
from unicodedata import normalize
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
#Initiallizing RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import Adam


#importing dataset
dataset = pd.read_csv("/kaggle/input/Covid_forecast.csv")
submission = dataset[["ForecastId", "ConfirmedCases", "Fatalities"]]
print(submission)
submission.to_csv("/kaggle/working/submission.csv", index=False)


#print(dataset)