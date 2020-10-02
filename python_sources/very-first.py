# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print("START")
data = pd.read_csv("../input/pik-competition-2018-v1/train.csv") 
y = data.price
predictors=["spalen", "vid_0", "vid_1", "vid_2", "month", "Class of object", "Number of rooms", "Fenced area", "Land area", "Input Groups", "Kindergarten", "School", "Polyclinic", "FOK" , "Playground","Car Wash","Pantry","Strollers","Conditioning","Ventilation","Elevator","The system of rubbish","CCTV","Underground parking","Yard without cars","Parking spaces","The area of prom. zones within a radius of 500 m","Area of the green zone within a radius of 500 m","TTK (km)","Sadoovoe(km)","To the underground on foot (km)","Metro stations from the ring"]
X = data[predictors]
model = DecisionTreeRegressor()
model.fit(X, y)

print("Making predictions for the following 5 houses:")
test = pd.read_csv("../input/pik-competition-2018-test-v1/test.csv") 
X1 = test[predictors]
print(X1.price.head())
print("The predictions are")
print(model.predict(X1.head()))


print("END")
# Any results you write to the current directory are saved as output.