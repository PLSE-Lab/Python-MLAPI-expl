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

data = read.csv("train_sample.csv")

attach(data)
data$Semana = as.factor(data$Semana)
data$Agencia_ID = as.factor(data$Agencia_ID)
data$Canal_ID = as.factor(data$Canal_ID)
data$Ruta_SAK = as.factor(data$Ruta_SAK)
data$Cliente_ID = as.factor(data$Cliente_ID)
data$Producto_ID = as.factor(data$Producto_ID)
sapply(data, class)

data[Demanda_uni_equil>125]$Demanda_uni_equil <- 125

fit <- lm(Demanda_uni_equil~Semana+Cliente_ID+Producto_ID,data=data)

summary(fit)