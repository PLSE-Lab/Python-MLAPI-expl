import pandas as pd
import numpy as np
import time    
import cv2
from tensorflow.keras.utils import Sequence

TRAIN_RANGE=200840
DATA_PATH = '../input/bengaliai-cv19/'
TRAIN=['train_image_data_0.parquet','train_image_data_1.parquet','train_image_data_2.parquet','train_image_data_3.parquet']
LABLES=pd.read_csv(DATA_PATH +  'train.csv')

class batch():

    def __init__(self,batch_size):
        self.batch_size=batch_size
        self.pre=0
        self.data=pd.read_parquet(DATA_PATH + 'train_image_data_0.parquet')
        self.len=len(self.data)
        self.index=0

    def set_next_df(self):
        if self.index + 1 < len(TRAIN):
            self.data=pd.read_parquet(DATA_PATH + TRAIN[self.index + 1])
            self.index += 1
        else:
            self.data=pd.read_parquet(DATA_PATH + TRAIN[0])
            self.index = 0
        self.pre=0
        self.len=len(self.data)


    def next_batch(self,get_as_array=False):
        if self.pre +  self.batch_size > self.len:
            if self.pre < self.len:
                x=self.data.iloc[self.pre  : ]
                len_of_x= len(x)
                self.set_next_df()
                x=x.append(self.data.iloc[self.pre : self.batch_size - len_of_x ] , ignore_index=True)
                self.pre = self.batch_size - len_of_x
                y=self.get_y(x)
                if get_as_array:
                    return self.get_asArray(x,y)
                return x,y
            else:
                    self.set_next_df()
        x=self.data.iloc[self.pre  :  self.pre +  self.batch_size]
        #y=self.label[self.pre :  self.pre +  self.batch_size]
        y=self.get_y(x)
        self.pre+=self.batch_size
        if get_as_array:
            return self.get_asArray(x,y)
        return x,y

    def get_y(self,data):
        l=len(data)
        d=data['image_id'].iloc[0]
        index=LABLES[LABLES['image_id']==d].index[0]
        if index + l > TRAIN_RANGE:
            y=LABLES.iloc[index:]
            len_of_y= len(y)
            y=y.append(LABLES[0: l - len_of_y ])
            return y
        y=LABLES.iloc[index:index+l]
        return y
    def get_asArray(self,x,y):
        x_val=x.drop('image_id',axis=1).values
        #y_val=y.drop(['image_id','grapheme'],axis=1).values
        y_val=y.drop(['image_id','grapheme','vowel_diacritic','consonant_diacritic'],axis=1).values
        return x_val,y_val


class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self,to_fit=True, batch_size=32, dim=(80, 80), n_classes=168):
     
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_classes = n_classes
        self.b=batch(batch_size)
        #self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(TRAIN_RANGE / self.batch_size))

    def __getitem__(self, index):
       
        X,Y=self.b.next_batch()

        if self.to_fit:
            return X, y
        else:
            return X

