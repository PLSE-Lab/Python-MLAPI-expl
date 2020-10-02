# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


import csv

class Data:
	datas=None
	labels=None
	idx=0
	def __init__(self):
		with open('../input/creditcard.csv','r') as file:
			reader=csv.reader(file)


			fake_num=0

			datas=[]
			labels=[]

			cnt=0
			for row in reader:
				data=[]
				label=[0]*2

				if cnt==0:
					cnt=1
					continue

				for i in range(len(row)):
					cell=float(row[i])

					if i==len(row)-1:
						if cell==1.0:
							fake_num+=1
							label[0]=1
						else:
							label[1]=1
						labels.append(label)
					else:
						data.append(cell)
				datas.append(data)
			del datas[0]
			del labels[0]

			self.datas=np.array(datas)
			self.labels=np.array(labels)

			print(fake_num)

data=Data()

# Any results you write to the current directory are saved as output.