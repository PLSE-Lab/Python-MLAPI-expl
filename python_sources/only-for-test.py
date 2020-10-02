# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
gatrain = pd.read_csv('../input/gender_age_train.csv')
gatest = pd.read_csv('../input/gender_age_test.csv')
gatrain.head(3)
app_labels = pd.read_csv('../input/app_labels.csv')
app_labels.head(3)
app_categories = pd.read_csv('../input/label_categories.csv')
app_categories.head(3)
phone_brand_device_model = pd.read_csv('../input/phone_brand_device_model.csv')
phone_brand_device_model.head(3)
app_events = pd.read_csv('../input/app_events.csv')
app_events.head(3)