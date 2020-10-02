"""
  So it's possible to embed Graphs in Google Maps.
  
  You can see an example of that here:
      https://storage.googleapis.com/montco-stats/elevator.html
      
  True, you have to kinda zoom in to see the maps.      

  I want to see if this is possible to do on Kaggle.


"""


import pandas as pd
import numpy as np
import datetime


import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)



# Read data 
d=pd.read_csv("../input/911.csv",
    header=0,names=['lat', 'lng','desc','zip','title','timeStamp','twp','addr','e'],
    dtype={'lat':str,'lng':str,'desc':str,'zip':str,
                  'title':str,'timeStamp':datetime.datetime,'twp':str,'addr':str,'e':int}, 
     )




# Set index
d.index = pd.DatetimeIndex(d.timeStamp)
d=d[(d.timeStamp >= "2016-01-01 00:00:00")]




from pytz import timezone
import pytz
eastern = timezone('US/Eastern')


import time
# now().strftime('%Y-%m-%d %H:%M:%S')
def nowf():
    return now().strftime('%Y-%m-%d %H:%M:%S')
def now():
    timezone='US/Eastern'
    native=datetime.datetime.now()
    if time.timezone == 0:
        return native.replace(tzinfo=pytz.utc).astimezone(pytz.timezone(timezone))
    else:
        return native



# Take a look at the following
#
# http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/


def mtext(text):
    (xmin,xmax,ymin,ymax)=plt.axis()
    xdist=xmax-xmin
    ydist=ymax-ymin  
    ypos=ymin-2*(ydist/8.0)
    plt.text(xmin,ypos,text,fontsize=8)
#    plt.savefig('junk.svg',format="svg", bbox_inches="tight")

def pltSetup(title):
  ax = plt.subplot(111)
  ax.spines["top"].set_visible(False)
  ax.spines["bottom"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.spines["left"].set_visible(False)
  # Ensure that the axis ticks only show up on the bottom and left of the plot.
  # Ticks on the right and top of the plot are generally unnecessary chartjunk.
  ax.get_xaxis().tick_bottom()
  ax.get_yaxis().tick_left()
  ax.spines["top"].set_visible(False)
  ax.spines["bottom"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.spines["left"].set_visible(False)
  ax.get_xaxis().tick_bottom()
  ax.get_yaxis().tick_left()
  ax.set_title(title,fontsize=10)
  plt.xticks(fontsize=8,rotation=45)
  plt.yticks(fontsize=7)



def createPivot(hr=24):
  d.timeStamp=pd.DatetimeIndex(d.timeStamp)
  t24=datetime.datetime.now() - datetime.timedelta(hours=hr)
  tz=d[(d.timeStamp >= t24)]
  g = tz.groupby(['title']).agg({'e':sum})
  h=g.sort_values(by='e',ascending=False).head(2)['e'].to_dict()
  tz=tz[tz['title'].isin(h.keys())]
  p=pd.pivot_table(tz, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)
  p.fillna(0, inplace=True)
  j=p.resample('24H',how='sum')  
  j.fillna(0, inplace=True)
#  j.index=j.index-pd.offsets.Hour(j.index.min().hour) - pd.offsets.Minute(j.index.min().minute) -pd.offsets.Second(j.index.min().second)
#  j.to_csv(file,index=True,header=True)
  j.to_csv('j.csv',index=True,header=True)
  jj=pd.read_csv("./j.csv")
  return (jj,j)

#(j,gj)=createPivot(hr=1000)

k=d.resample('24H',how='sum')  
k['timeStamp']=k.index
k=k[['timeStamp','e']]
k.sort_values(by=['timeStamp'],ascending=True,inplace=True)
k.drop(k.index[-1], inplace=True)  # remove last row
k.drop(k.index[0], inplace=True)  # remove first
j=k






# These are the "Tableau 20" colors as RGB.  
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
tableau20 = [(0, 255, 0), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]  

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.  
for i in range(len(tableau20)):  
    r, g, b = tableau20[i]  
    tableau20[i] = (r / 255., g / 255., b / 255.)  

def fixTime(x):
    try:
        return datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')
    except:
        try:
           return datetime.datetime.strptime(x,'%Y-%m-%d')
        except:
           return x
      
#    return int(datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S').hour)



tics=['ro', 'bs', 'g^','ro','bo','yo','r^','go','g--','ro','go','bo','r^']


def plotJ(j,file,srank=0,Y=-18):
  j.fillna(0, inplace=True)  
  timeHrs=[fixTime(x) for x in j.timeStamp.values]
  rank=srank
  for col in list(j.columns)[1::]:
    plt.plot(timeHrs,
             j[col].values,tics[rank],
             lw=0.9, color=tableau20[rank])
    rank+=1
  rank=srank
  for col in list(j.columns)[1::]:
    plt.plot(timeHrs,
             j[col].values,
             lw=0.9, color=tableau20[rank])
    rank+=1
#  time=utc_to_time(datetime.datetime.now())
#  plt.text(j.timeStamp.min(), Y, "Data source: http://montcoalert.org/"
#                  "\nAuthor: Mike Chirico (mchirico@gmail.com)"
#                  "\nTimeStamp:%s" % (time.strftime("%Y-%m-%d %H:%M:%S")), fontsize=7)
#  f=plt.gcf()
  #print(f.get_size_inches()) # [8.,6.]
  #f.set_size_inches(6,4.5)
  mtext('montcoalert.org\nAuthor: Mike Chirico mchirico@gmail.com\nTimestamp: %s'% (nowf()))
  plt.savefig(file+'.png',format="png", bbox_inches="tight")
  plt.savefig(file+'.svg',format="svg", bbox_inches="tight")
  plt.plot()  



plt.clf()
plt.cla()
pltSetup('All Incidents/Day')
plotJ(j,'dailyIncident')


plt.clf()
plt.cla()
pltSetup('All Incidents/Hour')
k=d.resample('1H',how='sum')  
k['timeStamp']=k.index
k=k[['timeStamp','e']]
k.sort_values(by=['timeStamp'],ascending=True,inplace=True)
k.drop(k.index[0:-172], inplace=True)  # remove all but last 172 hrs
plotJ(k,'hourlyIncident',4)


# Ten min
plt.clf()
plt.cla()
pltSetup('All Incidents/10min')
k=d.resample('10T',how='sum')  
k['timeStamp']=k.index
k=k[['timeStamp','e']]
k.sort_values(by=['timeStamp'],ascending=True,inplace=True)
k.drop(k.index[0:-60], inplace=True)  # remove all but last ...
plotJ(k,'tenMinIncident',5,-1)



# EMS: UNKNOWN MEDICAL EMERGENCY
plt.clf()
plt.cla()
pltSetup('EMS: UNKNOWN -  /24hr')
k=d[(d.title == 'EMS: UNKNOWN MEDICAL EMERGENCY')]
k.index=pd.DatetimeIndex(k.timeStamp)
k=k.resample('24H',how='sum')  
k['timeStamp']=k.index
k=k[['timeStamp','e']]
k.sort_values(by=['timeStamp'],ascending=True,inplace=True)
k.drop(k.index[0:-60], inplace=True)  # remove all but last ...
plotJ(k,'dailyIncidentEMSunknown',5,-1)


# EMS: UNKNOWN MEDICAL EMERGENCY  hour
plt.clf()
plt.cla()
pltSetup('EMS: UNKNOWN -  /hr')
k=d[(d.title == 'EMS: UNKNOWN MEDICAL EMERGENCY')]
k.index=pd.DatetimeIndex(k.timeStamp)
k=k.resample('60T',how='sum')  
k['timeStamp']=k.index
k=k[['timeStamp','e']]
k.sort_values(by=['timeStamp'],ascending=True,inplace=True)
k.drop(k.index[0:-60], inplace=True)  # remove all but last ...
plotJ(k,'hourIncidentEMSunknown',5,-1)


# EMS: UNKNOWN MEDICAL EMERGENCY  10/min
plt.clf()
plt.cla()
pltSetup('EMS: UNKNOWN -  /10min')
k=d[(d.title == 'EMS: UNKNOWN MEDICAL EMERGENCY')]
k.index=pd.DatetimeIndex(k.timeStamp)
k=k.resample('10T',how='sum')  
k['timeStamp']=k.index
k=k[['timeStamp','e']]
k.sort_values(by=['timeStamp'],ascending=True,inplace=True)
#k.drop(k.index[0:-6], inplace=True)  # remove all but last ...
plotJ(k,'tenminIncidentEMSunknown',5,-1)


# Helicopter Landing
plt.clf()
plt.cla()
pltSetup('Helicopter Landing  /120hr')
k=d[d.title.str.match(r'.*HELICOPTER.*')]
k.index=pd.DatetimeIndex(k.timeStamp)
k=k.resample('120H',how='sum')  
k['timeStamp']=k.index
k=k[['timeStamp','e']]
k.sort_values(by=['timeStamp'],ascending=True,inplace=True)
#k.drop(k.index[0:-6], inplace=True)  # remove all but last ...
plotJ(k,'t120helicopter',5,-1)


plt.clf()
plt.cla()
pltSetup('Helicopter Landing  /360hr')
k=d[d.title.str.match(r'.*HELICOPTER.*')]
k.index=pd.DatetimeIndex(k.timeStamp)
k=k.resample('360H',how='sum')  
k['timeStamp']=k.index
k=k[['timeStamp','e']]
k.sort_values(by=['timeStamp'],ascending=True,inplace=True)
#k.drop(k.index[0:-6], inplace=True)  # remove all but last ...
plotJ(k,'t360helicopter',5,-1)



# EMS: ANIMAL BITE
plt.clf()
plt.cla()
pltSetup('EMS: ANIMAL BITE  /120hr')
k=d[(d.title == 'EMS: ANIMAL BITE')]
k.index=pd.DatetimeIndex(k.timeStamp)
k=k.resample('120H',how='sum')  
k['timeStamp']=k.index
k=k[['timeStamp','e']]
k.sort_values(by=['timeStamp'],ascending=True,inplace=True)
#k.drop(k.index[0:-6], inplace=True)  # remove all but last ...
plotJ(k,'t120EmsAnimalBite',5,-1)


# EMS: ANIMAL BITE
plt.clf()
plt.cla()
pltSetup('EMS: ANIMAL BITE  /500hr')
k=d[(d.title == 'EMS: ANIMAL BITE')]
k.index=pd.DatetimeIndex(k.timeStamp)
k=k.resample('500H',how='sum')  
k['timeStamp']=k.index
k=k[['timeStamp','e']]
k.sort_values(by=['timeStamp'],ascending=True,inplace=True)
#k.drop(k.index[0:-6], inplace=True)  # remove all but last ...
plotJ(k,'fiveEmsAnimalBite',5,-1)




# EMS: DEHYDRATION Day
plt.clf()
plt.cla()
pltSetup('EMS: DEHYDRATION -  /24hr')
k=d[(d.title == 'EMS: DEHYDRATION')]
k.index=pd.DatetimeIndex(k.timeStamp)
k=k.resample('24H',how='sum')  
k['timeStamp']=k.index
k=k[['timeStamp','e']]
k.sort_values(by=['timeStamp'],ascending=True,inplace=True)
#k.drop(k.index[0:-6], inplace=True)  # remove all but last ...
plotJ(k,'dailyIncidentEMSDehydration',5,-1)

# EMS: DEHYDRATION  /72hr
plt.clf()
plt.cla()
pltSetup('EMS: DEHYDRATION -  /72hr')
k=d[(d.title == 'EMS: DEHYDRATION')]
k.index=pd.DatetimeIndex(k.timeStamp)
k.index=pd.DatetimeIndex(k.timeStamp)
k=k.resample('72H',how='sum')  
k['timeStamp']=k.index
k=k[['timeStamp','e']]
k.sort_values(by=['timeStamp'],ascending=True,inplace=True)
#k.drop(k.index[0:-60], inplace=True)  # remove all but last ...
plotJ(k,'seventy2HrEMSDehydration',5,-1)

# EMS: DEHYDRATION  /72hr
plt.clf()
plt.cla()
pltSetup('EMS: DEHYDRATION -  /300hr')
k=d[(d.title == 'EMS: DEHYDRATION')]
k.index=pd.DatetimeIndex(k.timeStamp)
k.index=pd.DatetimeIndex(k.timeStamp)
k=k.resample('300H',how='sum')  
k['timeStamp']=k.index
k=k[['timeStamp','e']]
k.sort_values(by=['timeStamp'],ascending=True,inplace=True)
#k.drop(k.index[0:-60], inplace=True)  # remove all but last ...
plotJ(k,'three100HrEMSDehydration',5,-1)




# EMS: CARDIAC EMERGENCY Day
plt.clf()
plt.cla()
pltSetup('EMS: CARDIAC EMERGENCY -  /24hr')
k=d[(d.title == 'EMS: CARDIAC EMERGENCY')]
k.index=pd.DatetimeIndex(k.timeStamp)
k=k.resample('24H',how='sum')  
k['timeStamp']=k.index
k=k[['timeStamp','e']]
k.sort_values(by=['timeStamp'],ascending=True,inplace=True)
#k.drop(k.index[0:-6], inplace=True)  # remove all but last ...
plotJ(k,'dailyIncidentEMSCardiac',5,-1)

# EMS: CARDIAC EMERGENCY 6 hour
plt.clf()
plt.cla()
pltSetup('EMS: CARDIAC EMERGENCY -  /6hr')
k=d[(d.title == 'EMS: CARDIAC EMERGENCY')]
k.index=pd.DatetimeIndex(k.timeStamp)
k=k.resample('6H',how='sum')  
k['timeStamp']=k.index
k=k[['timeStamp','e']]
k.sort_values(by=['timeStamp'],ascending=True,inplace=True)
#k.drop(k.index[0:-6], inplace=True)  # remove all but last ...
plotJ(k,'hourlyEMSCardiac',5,-1)

# EMS: CARDIAC EMERGENCY  /72hr
plt.clf()
plt.cla()
pltSetup('EMS: CARDIAC EMERGENCY -  /72hr')
k=d[(d.title == 'EMS: CARDIAC EMERGENCY')]
k.index=pd.DatetimeIndex(k.timeStamp)
k.index=pd.DatetimeIndex(k.timeStamp)
k=k.resample('72H',how='sum')  
k['timeStamp']=k.index
k=k[['timeStamp','e']]
k.sort_values(by=['timeStamp'],ascending=True,inplace=True)
#k.drop(k.index[0:-60], inplace=True)  # remove all but last ...
plotJ(k,'tenMinEMSCardiac',5,-1)




# Overdose Day
plt.clf()
plt.cla()
pltSetup('EMS: OVERDOSE -  /24hr')
k=d[(d.title == 'EMS: OVERDOSE')]
k.index=pd.DatetimeIndex(k.timeStamp)
k=k.resample('24H',how='sum')  
k['timeStamp']=k.index
k=k[['timeStamp','e']]
k.sort_values(by=['timeStamp'],ascending=True,inplace=True)
#k.drop(k.index[0:-6], inplace=True)  # remove all but last ...
plotJ(k,'dailyIncidentEMSOverdose',5,-1)


# Overdose  hourly
plt.clf()
plt.cla()
pltSetup('EMS: OVERDOSE -  /hr')
k=d[(d.title == 'EMS: OVERDOSE')]
k.index=pd.DatetimeIndex(k.timeStamp)
k=k.resample('60T',how='sum')  
k['timeStamp']=k.index
k=k[['timeStamp','e']]
k.sort_values(by=['timeStamp'],ascending=True,inplace=True)
#k.drop(k.index[0:-6], inplace=True)  # remove all but last ...
plotJ(k,'hourlyEMSOverdose',5,-1)



# Overdose  10/min
plt.clf()
plt.cla()
pltSetup('EMS: OVERDOSE -  /10min')
k=d[(d.title == 'EMS: OVERDOSE')]
k.index=pd.DatetimeIndex(k.timeStamp)
k=k.resample('10T',how='sum')  
k['timeStamp']=k.index
k=k[['timeStamp','e']]
k.sort_values(by=['timeStamp'],ascending=True,inplace=True)
#k.drop(k.index[0:-60], inplace=True)  # remove all but last ...
plotJ(k,'tenMinEMSOverdose',5,-1)




# Traffic only
plt.clf()
plt.cla()
pltSetup('Traffic: VEHICLE ACCIDENT -  /10min')
k=d[(d.title == 'Traffic: VEHICLE ACCIDENT -')]
k.index=pd.DatetimeIndex(k.timeStamp)
k=k.resample('10T',how='sum')  
k['timeStamp']=k.index
k=k[['timeStamp','e']]
k.sort_values(by=['timeStamp'],ascending=True,inplace=True)
k.drop(k.index[0:-60], inplace=True)  # remove all but last ...
plotJ(k,'tenMinTrafficVehicleAccident',5,-1)

plt.clf()
plt.cla()
pltSetup('Traffic: VEHICLE ACCIDENT -  /Hour')
k=d[(d.title == 'Traffic: VEHICLE ACCIDENT -')]
k.index=pd.DatetimeIndex(k.timeStamp)
k=k.resample('60T',how='sum')  
k['timeStamp']=k.index
k=k[['timeStamp','e']]
k.sort_values(by=['timeStamp'],ascending=True,inplace=True)
k.drop(k.index[0:-60], inplace=True)  # remove all but last ...
plotJ(k,'hourlyTrafficVehicleAccident',5,-1)


# EMS: VEHICLE ACCIDENT
plt.clf()
plt.cla()
pltSetup('EMS: VEHICLE ACCIDENT  /10min')
k=d[(d.title == 'EMS: VEHICLE ACCIDENT')]
k.index=pd.DatetimeIndex(k.timeStamp)
k=k.resample('10T',how='sum')  
k['timeStamp']=k.index
k=k[['timeStamp','e']]
k.sort_values(by=['timeStamp'],ascending=True,inplace=True)
k.drop(k.index[0:-60], inplace=True)  # remove all but last ...
plotJ(k,'tenMinEMSVehicleAccident',5,-1)

plt.clf()
plt.cla()
pltSetup('EMS: VEHICLE ACCIDENT  /Hour')
k=d[(d.title == 'EMS: VEHICLE ACCIDENT')]
k.index=pd.DatetimeIndex(k.timeStamp)
k=k.resample('60T',how='sum')  
k['timeStamp']=k.index
k=k[['timeStamp','e']]
k.sort_values(by=['timeStamp'],ascending=True,inplace=True)
k.drop(k.index[0:-60], inplace=True)  # remove all but last ...
plotJ(k,'hourlyEMSVehicleAccident',5,-1)


# Vomit
plt.clf()
plt.cla()
pltSetup('EMS: NAUSEA/VOMITING  /day')
k=d[(d.title == 'EMS: NAUSEA/VOMITING')]
k.index=pd.DatetimeIndex(k.timeStamp)
k=k.resample('24H',how='sum')  
k['timeStamp']=k.index
k=k[['timeStamp','e']]
k.sort_values(by=['timeStamp'],ascending=True,inplace=True)
#k.drop(k.index[0:-60], inplace=True)  # remove all but last ...
plotJ(k,'dailyEMSNauseaVomiting',5,-1)


plt.clf()
plt.cla()
pltSetup('EMS: NAUSEA/VOMITING  /72 hrs')
k=d[(d.title == 'EMS: NAUSEA/VOMITING')]
k.index=pd.DatetimeIndex(k.timeStamp)
k=k.resample('72H',how='sum')  
k['timeStamp']=k.index
k=k[['timeStamp','e']]
k.sort_values(by=['timeStamp'],ascending=True,inplace=True)
#k.drop(k.index[0:-60], inplace=True)  # remove all but last ...
plotJ(k,'v72hrEMSNauseaVomiting',5,-1)


plt.clf()
plt.cla()
pltSetup('Fire: Electrical Outside\nNumber of Calls  Within 72 hrs')
#k=d[(d.title == 'EMS: NAUSEA/VOMITING')]
#k=d[d.title.str.match(r'Traffic.*ELECTRICAL.*')]
k=d[(d.title == 'Fire: ELECTRICAL FIRE OUTSIDE')]
k.index=pd.DatetimeIndex(k.timeStamp)
k=k.resample('72H',how='sum')  
k['timeStamp']=k.index
k=k[['timeStamp','e']]
k.sort_values(by=['timeStamp'],ascending=True,inplace=True)
#k.drop(k.index[0:-60], inplace=True)  # remove all but last ...
plotJ(k,'v72hrFireElectricalOutside',5,-1)


# Electrical Fire
plt.clf()
plt.cla()
pltSetup('Fire: Electrical Outside\nNumber of Calls /10min')
k=d[(d.title == 'Fire: ELECTRICAL FIRE OUTSIDE')]
k.index=pd.DatetimeIndex(k.timeStamp)
k=k.resample('10T',how='sum')  
k['timeStamp']=k.index
k=k[['timeStamp','e']]
k.sort_values(by=['timeStamp'],ascending=True,inplace=True)
k.drop(k.index[0:-60], inplace=True)  # remove all but last ...
plotJ(k,'tenMinFireElectricalOutside',5,-1)



