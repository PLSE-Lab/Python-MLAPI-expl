# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
getwd()
setwd("C:/Users/User")
data=read.csv(file.choose())
data
summary(data)

class(data)
data1=data[c(2,4,5,6,7,8,11,12,15,16,17,18,29,190,198,206,214,222,230,238,246,254,262,294,302,310,342,366,491,499,503,587,783,786)]
data1
plot(data1$STATCD,data1$OVERALL_LI)
summary(data1)
n=NROW(data)
trainindex=sample(1:n, size=round(0.7*n), replace = FALSE)
train=data1[trainindex,]
test=data1[-trainindex,]
boxplot(train)
cor(train)
#library(corrplot)
#corrplot(cor(train),method="number",type="upper")
rel=lm(OVERALL_LI~FEMALE_LIT+MALE_LIT+GROWTHRATE+SEXRATIO, data=train)
summary(rel)
rel1=lm(OVERALL_LI~GROWTHRATE+SEXRATIO+FEMALE_LIT+MALE_LIT, data=test)
summary(rel1)
a=data.frame(GROWTHRATE=23.71,SEXRATIO=883,FEMALE_LIT=58.01,MALE_LIT=78.26)
predict(rel1,a)
train2=train[c(11,12,16,17)]
test2=test[c(11,12,16,17)]
plot(train2)
plot(test2)

