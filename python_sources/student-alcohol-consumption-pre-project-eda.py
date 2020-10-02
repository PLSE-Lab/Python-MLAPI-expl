import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt


df = pd.read_csv('../input/student-mat.csv')
df.info()
df.describe()

sns.heatmap(df.isna())

plt.figure(figsize=(15,15))
sns.heatmap(df.corr(), annot = True)

df['Aalc'] = df['Dalc'] + df['Walc']
sns.countplot(x="Aalc", data=df)
plt.ylabel('Number of Students')
plt.xlabel('Weekly alcohol consumption')

prc=df[['Aalc', 'sex']].groupby(['Aalc']).count().apply(lambda x:100 * x / float(x.sum()))


ave = sum(df.G3)/float(len(df))
df['average'] = ['above average' if i > ave else 'under average' for i in df.G3]
sns.swarmplot(x='Aalc', y = 'G3', hue = 'average',data= df)

sns.catplot(x="sex", kind="count", data=df)

sns.catplot(x="studytime", kind="count", data=df)

sns.catplot(x="address", kind="count",hue = "traveltime",data=df)
sns.catplot(x="address", kind="count",hue = "sex",data=df)

sns.distplot(df[(df.address == 'U')]["absences"],color='orange')
sns.distplot(df[(df.address == 'R')]["absences"],color='orange')

sns.boxplot(y="famsize", x="G3", data = df , orient="h", palette = 'winter')

sns.jointplot(x="age", y="G3", data = df[(df.paid == 'no')],kind="kde")

sns.factorplot(x="Walc", y="health", hue="sex", data=df)


def change_size(size):
    if 2 == size:
        return(2)
    elif 1 == size:
        return(1)
    else:
        return None

df["New"] = df["Aalc"].map(change_size)
