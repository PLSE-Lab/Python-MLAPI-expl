# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
import os

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
app = pd.read_csv('../input/app_events.csv')
events = pd.read_csv('../input/events.csv')
pbd = pd.read_csv('../input/phone_brand_device_model.csv')
train = pd.read_csv('../input/gender_age_train.csv')
test = pd.read_csv('../input/gender_age_test.csv')


#event_id
print('event_id number in app_events: ',len(set(list(app['event_id']))))
print('event_id number in events: ',len(set(list(events['event_id']))))

#device_id
print('device_id number in phone_brand_device_model: ',len(set(list(pbd['device_id']))))
print('device_id number in events: ',len(set(list(events['device_id']))))
print('device_id numberr in train: ',len(set(list(train['device_id']))))
print('device_id number in test: ', len(set(list(test['device_id']))))
print('device_id number in train & test : ' ,len(set(list(train['device_id'])+list(test['device_id']))))
#brand and device_model
print('number of brand in pbd:',len(set(list(pbd['phone_brand']))))
print('number of device_model in pbd: ',len(set(list(pbd['device_model']))))



