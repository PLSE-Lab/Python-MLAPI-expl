#!/usr/bin/env python
# coding: utf-8

# "use resolution L"  

# In[ ]:


###########################################################
#By: Michael Lemacks
#
#
#
###########################################################
# This just lets the output of the following code samples
#  display inline on this page, at an appropriate size.
from pylab import rcParams
#test save

rcParams['figure.figsize'] = (8,6)
import numpy as np  
import pandas as pd

# Module 1
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

terror = pd.read_csv("../input/globalterrorismdb_0616dist.csv", encoding='ISO-8859-1', dtype = "object")


# In[ ]:



plt.figure(figsize=(15,8))
southamerica = terror[terror["region_txt"] == "South America"]

SA = Basemap(projection='mill', llcrnrlat = -60, urcrnrlat = 25, llcrnrlon = -100, urcrnrlon = -20, resolution = 'h')
SA.drawcoastlines()
SA.drawcountries()
SA.drawstates()

x, y = SA(list(southamerica["longitude"].astype("float")), list(southamerica["latitude"].astype(float)))
SA.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "darkgreen")

plt.title('Terror Attacks on South America (1970-2015)')
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
southamerica6 = terror[terror["weaptype1_txt"] == "Explosives/Bombs/Dynamite"]
southamerica8 = terror[terror["weaptype1_txt"] == "Incendiary"]
southamerica5 = terror[terror["weaptype1_txt"] == "Firearms"]

SA = Basemap(projection='mill', llcrnrlat = -60, urcrnrlat = 25, llcrnrlon = -100, urcrnrlon = -20, resolution = 'l')
SA.drawcoastlines()
SA.drawcountries()
SA.drawstates()

x, y = SA(list(southamerica6["longitude"].astype("float")), list(southamerica6["latitude"].astype(float)))
NA.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "yellow")

x, y = SA(list(southamerica8["longitude"].astype("float")), list(southamerica8["latitude"].astype(float)))
NA.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "red")

x, y = SA(list(southamerica5["longitude"].astype("float")), list(southamerica5["latitude"].astype(float)))
NA.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "green")

plt.title('Terror Attacks by Weapon Type on South America 1970-2015')
plt.show()


# In[ ]:


weaptype = terror[terror["region_txt"]=='South America'].groupby('weaptype1_txt').size().order(ascending=False).head(10).to_frame(name = 'count').reset_index()

weaptype


# In[ ]:


plt.figure(figsize=(12,8))
northamerica6 = terror[terror["weaptype1_txt"] == "Explosives/Bombs/Dynamite"]
northamerica8 = terror[terror["weaptype1_txt"] == "Incendiary"]
northamerica5 = terror[terror["weaptype1_txt"] == "Firearms"]

NA = Basemap(projection='mill', llcrnrlat = 0, urcrnrlat = 75, llcrnrlon = -170, urcrnrlon = -55, resolution = 'h')
NA.drawcoastlines()
NA.drawcountries()
NA.drawstates()

x, y = NA(list(northamerica6["longitude"].astype("float")), list(northamerica6["latitude"].astype(float)))
NA.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "yellow")

x, y = NA(list(northamerica8["longitude"].astype("float")), list(northamerica8["latitude"].astype(float)))
NA.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "red")

x, y = NA(list(northamerica5["longitude"].astype("float")), list(northamerica5["latitude"].astype(float)))
NA.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "green")

plt.title('Terror Attacks by Weapon Type on North America 1970-2015')
plt.show()


# In[ ]:


weaptype = terror[terror["country_txt"]=='United States'].groupby('weaptype1_txt').size().order(ascending=False).head(10).to_frame(name = 'count').reset_index()

weaptype


# In[ ]:


weaptype1 = terror[terror['weaptype1_txt']=="Incendiary"].groupby('weaptype1_txt').size().order(ascending=False).head(10).to_frame(name = 'count').reset_index()
weaptype1


# In[ ]:


plt.figure(figsize=(12,8))
northamerica = terror[terror["region_txt"] == "North America"]

NA = Basemap(projection='mill', llcrnrlat = 0, urcrnrlat = 75, llcrnrlon = -170, urcrnrlon = -55, resolution = 'h')
NA.drawcoastlines()
NA.drawcountries()
NA.drawstates()

x, y = NA(list(northamerica["longitude"].astype("float")), list(northamerica["latitude"].astype(float)))
NA.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "lightblue")

plt.title('Terror Attacks on North America 1970-2015')
plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
centralamerica = terror[terror["region_txt"] == "Central America & Caribbean"]

CA = Basemap(projection='mill', llcrnrlat = 0, urcrnrlat = 30, llcrnrlon = -105, urcrnrlon = -60, resolution = 'h')
CA.drawcoastlines()
CA.drawcountries()
CA.drawstates()

x, y = CA(list(centralamerica["longitude"].astype("float")), list(centralamerica["latitude"].astype(float)))
CA.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "blue")

plt.title('Terror Attacks on Central America & Caribbean (1970-2015)')
plt.show()


# In[ ]:





# In[ ]:


weaptype = terror[terror["region_txt"]=='Central America & Caribbean'].groupby('weaptype1_txt').size().order(ascending=False).head(10).to_frame(name = 'count').reset_index()

weaptype


# In[ ]:


plt.figure(figsize=(15,8))
middleeast = terror[terror["region_txt"] == "Middle East & North Africa"]

ME = Basemap(projection='mill', llcrnrlat = 0, urcrnrlat = 60, llcrnrlon = -35, urcrnrlon = 65, resolution = 'h')
ME.drawcoastlines()
ME.drawcountries()
ME.drawstates()

x, y = ME(list(middleeast["longitude"].astype("float")), list(middleeast["latitude"].astype(float)))
ME.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "#FF0000")

plt.title('Terror Attacks on Middle East & North Africa (1970-2015)')
plt.show()


# In[ ]:


weaptype = terror[terror["region_txt"]=='Middle East & North Africa'].groupby('weaptype1_txt').size().order(ascending=False).head(10).to_frame(name = 'count').reset_index()

weaptype


# In[ ]:


plt.figure(figsize=(15,8))
subsaharan = terror[terror["region_txt"] == "Sub-Saharan Africa"]

SSA = Basemap(projection='mill', llcrnrlat = -45, urcrnrlat = 60, llcrnrlon = -35, urcrnrlon = 65, resolution = 'h')
SSA.drawcoastlines()
SSA.drawcountries()
SSA.drawstates()

x, y = SSA(list(subsaharan["longitude"].astype("float")), list(subsaharan["latitude"].astype(float)))
SSA.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "#FFFF00")

plt.title('Terror Attacks on Sub-Saharan Africa (1970-2015)')
plt.show()


# In[ ]:


weaptype = terror[terror["region_txt"]=='Sub-Saharan Africa'].groupby('weaptype1_txt').size().order(ascending=False).head(10).to_frame(name = 'count').reset_index()

weaptype


# In[ ]:


plt.figure(figsize=(15,8))
europe = terror[terror["region_txt"].isin(["Eastern Europe", "Western Europe"])]

EU = Basemap(projection='mill', llcrnrlat = 10, urcrnrlat = 75, llcrnrlon = -15, urcrnrlon = 70, resolution = 'h')
EU.drawcoastlines()
EU.drawcountries()
EU.drawstates()

x, y = EU(list(europe["longitude"].astype("float")), list(europe["latitude"].astype(float)))
EU.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "#66FF00")

plt.title('Terror Attacks on Europe (1970-2015)')
plt.show()


# In[ ]:


weaptype = terror[terror["region_txt"]=='Western Europe'].groupby('weaptype1_txt').size().order(ascending=False).head(10).to_frame(name = 'count').reset_index()

weaptype


# In[ ]:


plt.figure(figsize=(15,8))

m = Basemap(projection = 'mill', llcrnrlat = -80, urcrnrlat = 80, llcrnrlon = -180, urcrnrlon = 180, resolution = 'h')
m.drawcoastlines()
m.drawcountries()

# x, y = m(list(asia["longitude"].astype("float")), list(asia["latitude"].astype(float)))
# m.plot(x, y, "go", markersize = 6, alpha = 0.8, color = "#0000FF", label = "Asia")

x, y = m(list(europe["longitude"].astype("float")), list(europe["latitude"].astype(float)))
m.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "#66FF00", label = "Europe")

x, y = m(list(subsaharan["longitude"].astype("float")), list(subsaharan["latitude"].astype(float)))
m.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "#FFFF00", label = "SubSaharan Africa")

x, y = m(list(middleeast["longitude"].astype("float")), list(middleeast["latitude"].astype(float)))
m.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "#FF0000", label = "Middle East")

x, y = m(list(centralamerica["longitude"].astype("float")), list(centralamerica["latitude"].astype(float)))
m.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "blue", label = "Central America")

x, y = m(list(northamerica["longitude"].astype("float")), list(northamerica["latitude"].astype(float)))
m.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "darkblue", label = "North America")

x, y = m(list(southamerica["longitude"].astype("float")), list(southamerica["latitude"].astype(float)))
m.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "darkgreen", label = "South America")


plt.title('Global Terror Attacks (1970-2015)')
plt.legend()
plt.show()


# In[ ]:


groups = terror[terror["gname"]!="Unknown"].groupby('gname').size().order(ascending=False).head(10).to_frame(name = 'count').reset_index()

groups


# In[ ]:


plt.figure(figsize=(15,8))

taliban = terror[terror["gname"] == "Taliban"]
shiningpath = terror[terror["gname"] == "Shining Path (SL)"]
fmln = terror[terror["gname"] == "Farabundo Marti National Liberation Front (FMLN)"]
isil = terror[terror["gname"] == "Islamic State of Iraq and the Levant (ISIL)"]
ira = terror[terror["gname"] == "Irish Republican Army (IRA)"]
farc = terror[terror["gname"] == "Revolutionary Armed Forces of Colombia (FARC)"]
npa = terror[terror["gname"] == "New People's Army (NPA)"]
alsh = terror[terror["gname"] == "Al-Shabaab"]
eta = terror[terror["gname"] == "Basque Fatherland and Freedom (ETA)"]
boko = terror[terror["gname"] == "Boko Haram"]

m = Basemap(projection = 'mill', llcrnrlat = -80, urcrnrlat = 80, llcrnrlon = -180, urcrnrlon = 180, resolution = 'h')
m.drawcoastlines()
m.drawcountries()

x, y = m(list(taliban["longitude"].astype("float")), list(taliban["latitude"].astype(float)))
m.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "#FF0000", label = "Taliban")

x, y = m(list(shiningpath["longitude"].astype("float")), list(shiningpath["latitude"].astype(float)))
m.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "#66FF00", label = "Shining Path")

x, y = m(list(fmln["longitude"].astype("float")), list(fmln["latitude"].astype(float)))
m.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "#FFFF00", label = "FMLN")

x, y = m(list(isil["longitude"].astype("float")), list(isil["latitude"].astype(float)))
m.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "#FF3366", label = "ISIL")

x, y = m(list(ira["longitude"].astype("float")), list(ira["latitude"].astype(float)))
m.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "blue", label = "IRA")

x, y = m(list(farc["longitude"].astype("float")), list(farc["latitude"].astype(float)))
m.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "#CC6666", label = "FARC")

x, y = m(list(npa["longitude"].astype("float")), list(npa["latitude"].astype(float)))
m.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "darkgreen", label = "NPA")

x, y = m(list(alsh["longitude"].astype("float")), list(alsh["latitude"].astype(float)))
m.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "#F0E442", label = "Al-Shabaab")

x, y = m(list(eta["longitude"].astype("float")), list(eta["latitude"].astype(float)))
m.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "#D55E00", label = "ETA")

x, y = m(list(boko["longitude"].astype("float")), list(boko["latitude"].astype(float)))
m.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "#CC79A7", label = "Boko Haram")

plt.title('Global Terror Attacks (1970-2015) by Top 10 Terror Groups')
plt.legend()
plt.show()

