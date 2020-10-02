# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

trainview = pd.read_csv("../input/trainView.csv",
  header=0,names=['train_id','status','next_station','service','dest','lon','lat','source',
  'track_change','track','date','timeStamp0','timeStamp1'],
  dtype={'train_id':str,'status':str,'next_station':str,'service':str,'dest':str,
  'lon':str,'lat':str,'source':str,'track_change':str,'track':str,
  'date':str,
  'timeStamp0':datetime.datetime,'timeStamp1':datetime.datetime},
             parse_dates=['timeStamp0','timeStamp1'],date_parser=dateparse)

df = pd.DataFrame(trainview)

             
# Any results you write to the current directory are saved as output.