# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime,date
import matplotlib.pyplot as plt
import json

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
retail=pd.read_csv("../input/ign.csv")
release_date=retail.apply(lambda x:datetime.strptime("%s/%s/%s"%(x["release_year"],x["release_month"],x["release_day"]),"%Y/%m/%d"),axis=1)
retail["release_date"]=["%d-%d-%d"%(d.year,d.month,d.day) for d in release_date]
retailSorted=retail.sort_values(["release_date"],ascending=True)
platforms=retailSorted["platform"]
geners=retailSorted["genre"]
data=retailSorted["score"]
uplatforms=platforms.unique()
ugeners=geners.unique()
boolBgnPlatforms=platforms.duplicated()
boolEndPlatforms=platforms.duplicated("last")
beginPlatforms=retailSorted[~boolBgnPlatforms]
endPlatforms=retailSorted[~boolEndPlatforms]
final={
	"platforms":{},
	"genre":{},
	"begin":{},
	"end":{}
}

final["begin"]={
	"platforms":{
		"platforms":beginPlatforms["platform"].tolist(),
		"year":beginPlatforms["release_year"].tolist()
	}
}
final["end"]={
	"platforms":{
		"platforms":endPlatforms["platform"].tolist(),
		"year":endPlatforms["release_year"].tolist()
	}
}
def totalBinA(col_a,col_b,b_unique):
	out={}
	for b in b_unique:
		if(~pd.isnull(b)):
			out[b]=col_a[col_b==b].size
	return out

#print data[platforms=="iPad"]
for plat in uplatforms:
	platform=data[platforms==plat]
	#print "Games in ",plat," :",platform.size ," avg rview:",platform.mean()
	final["platforms"][plat]={
		"name":plat,
		"games":platform.size,
		"Avg Review":platform.mean(),
		"gener":totalBinA(platform,geners,ugeners)
	}
for gen in ugeners:
	gener=data[geners==gen]
	#print "Games in ",plat," :",platform.size ," avg rview:",platform.mean()
	final["genre"][gen]={
		"name":gen,
		"games":gener.size,
		"Avg Review":gener.mean(),
		"platform":totalBinA(gener,platforms,uplatforms)
	}
#print dumps(final)

def getMean(data,intervel):
	mean=[]
	for i in range(0,data.size/intervel):
		x=i*intervel
		if(x+intervel<=data.size):
			mean.append(pd.Series(data[x:x+intervel]).mean())
	return mean
def getReleasesByType(typ):
    x=np.array([index for index,val in retailSorted.groupby(typ).size().iteritems()]).tolist()
    y=np.array([val for index,val in retailSorted.groupby(typ).size().iteritems()]).tolist()
    return {"x":x,"y":y}
releaseYear=getReleasesByType("release_year")
releaseMonth=getReleasesByType("release_month")
releaseDay=getReleasesByType("release_day")
releaseDate=getReleasesByType("release_date")
plt.figure(figsize=(17,8))
#retailSorted.groupby().size("release_month").plot(kind="bar",c="green")
#retailSorted.groupby().size().plot(c=G)
#retailSorted.groupby().size().plot(c=G)
#retailSorted.groupby().size().plot(c=G)


# Any results you write to the current directory are saved as output.