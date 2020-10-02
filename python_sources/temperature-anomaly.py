import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../input/ghcn-m-v1.csv', header=0)
test_df = df[(df['year']==2015) & (df['month']==7)]
test_df = test_df.replace([-9999],[0])
             
lat = np.linspace(87.5,-87.5,36) # np.arange(2.5,180,5)-90
lon = np.linspace(-177.5,177.5,72)
labels = ['150 W','100 W','50 W', '0', '50 E','100 E', '150 E']
labels2 = ['40 N','20 N',0,'20 S','40 S']

temp_anomaly = test_df.values
temp_anomaly = temp_anomaly[:,3:] #temp anomaly matrix

temp_anomaly= temp_anomaly/100
vmin = np.min(temp_anomaly)
vmax = np.max(temp_anomaly)

lon_v, lat_h = np.meshgrid(lon,lat)
cm = plt.get_cmap('jet')
cp = plt.contourf(lon_v,lat_h,temp_anomaly,
                  cmap=cm, 
                  levels=np.linspace(-4,4,50))
                  #levels=np.linspace(math.ceil(vmin),math.ceil(vmax),50))
cb=plt.colorbar(cp, orientation='horizontal', 
                shrink=0.8,
                ticks=[-4,0,4])
plt.ylim(-49,50)
plt.xticks([-150,-100,-50,0,50,100,150],labels)
plt.yticks([40,20,0,-20,-40],labels2)
plt.title('Temperature anomaly (degree Celsius)_ 2015 July')
plt.hlines(0,-177.5,177.5,alpha=0.1)
plt.show()