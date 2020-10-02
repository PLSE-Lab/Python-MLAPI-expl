# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('../input/hotel-booking-demand/hotel_bookings.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")

print(data["is_canceled" == 1 ].shape(0) /data.shape[0])


# Any results you write to the current directory are saved as output.