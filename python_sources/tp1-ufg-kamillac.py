# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import sklearn.linear_model as lr
import sklearn.metrics as mtr
import plotly.plotly as py
import plotly.graph_objs as go
 
dataset = pd.read_csv("../input/auto-mpg.csv", usecols = ["weight", "horsepower", "mpg"]) #Selecionando os dados necessários no programa
datasetauto = dataset[dataset['horsepower'] != '?']

datasetauto = datasetauto.astype(float)

x = datasetauto[['weight', 'horsepower']]
y = datasetauto[['mpg']]

#x = datasetauto[datasetauto[['weight']] * 0.4535923]
#x = datasetauto[datasetauto[['horsepower']] * 0.4251437]

x = datasetauto[['weight', 'horsepower']].values
y = datasetauto[['mpg']].values


m = x.shape[0]
n = x.shape[1]

mtreinamento = round(m * 0.7)
Xtreinamento = x[:mtreinamento]
Ytreinamento = y[:mtreinamento]

Xteste = x[-mtreinamento:]
Yteste = y[-mtreinamento:]

reglinear = lr.LinearRegression()
reglinear.fit(Xtreinamento, Ytreinamento)
t0 = reglinear.intercept_
t1 = reglinear.coef_

Ytesteprevisto = reglinear.predict(Xteste)

errotreinamento = mtr.mean_squared_error(Ytreinamentoprevisto, Ytreinamento)
print("erro", errotreinamento)

plt.scatter(Xteste[:,[0]], Yteste)
plt.xlabel('Peso (Dataset Validação)')
plt.ylabel('Autonomia km/h (Dataset Validação)')
plt.scatter(Xteste[:,[0]], Ytesteprevisto)
plt.show()

#Código do gráfico 3d sem aplicação nos dados do programa

datasetauto = [
    go.Surface(
        z = datasetauto.as_matrix()
    )
]

layout = go.Layout(
    title = 'Autonomia encontrada a partir do peso e cavalos do veículo',
    autosize = False,
    width = 10000,
    height = 10000,
    margin = dict(
        l = 100,
        r = 100,
        b = 100,
        t = 100
    )
)

fig = go.Figure(datasetauto = datasetauto, layout = layout)
py.iplot(fig, filename = 'Gráfico 3D')


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.