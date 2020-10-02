import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
#to read playstore file
data = pd.read_csv('../input/googleplaystore.csv')
print(data.describe())
data = data.set_index('App')#to change index
data = data.fillna(0)#to remove all the nan value
data['Rating'].replace(to_replace=19,value=1.9,inplace=True)#to change the garbage value in dataset
data['Installs'].replace(to_replace='0',value='6+',inplace=True)#same as prev
data['Installs'].replace(to_replace='Free',value='5+',inplace=True)#same as prev
data = data.groupby('Installs')['Rating'].mean()#groupby no of downloads 
data = pd.DataFrame(data)
rat = data['Rating']#to store all the ratings to plot
v = list(rat)
#print(data)
d1 = data.index#to remove + and , in Installs
dtl = []
#whole loop which removes '+' and ',' in Installs column and convert it into int dtype 
for i in range(len(d1)):
    dt1 = d1[i]
    l = len(dt1)
    dt1 = dt1[:l-1]
    d=''
    for j in dt1:
        if j==',':
            continue
        else:
            d+=j
    dt1 = int(d)
    dtl.append(dt1)
inst = dtl#to plot store the no of downloads
d2 = pd.DataFrame(d1,index=inst,columns=['Installs','Rating'])
d2['Rating']=v#creating to store the rating in column
d2 = d2.sort_index(axis=0,ascending=True)
inst = d2.index
rat = d2['Rating']
rat = np.array(rat)
print(d2)
inst1 = np.array(inst,dtype = str)
plt.xlabel('NO OF DOWNLOADS')
plt.ylabel('RATINGS')
plt.title('RAT VS DOWN')
plt.bar(inst1,rat,color='blue')
plt.show()


#FROM THE GRAPH IT IS CLEAR THAT THE GRAPH IS NEGATIVELY SKEWED . HENCE, NO OF INSTALLS WILL INCREASE THE RATING

