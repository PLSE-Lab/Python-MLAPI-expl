#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import matplotlib.colors
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)


# # Exploratory Data Analysis

# In[ ]:


df=pd.read_csv("/kaggle/input/open-exoplanet-catalogue/oec.csv")


# **The dataset contains 3584 planets and 25 columns**

# In[ ]:


df.head()


# In[ ]:


df.describe()


# **Let's group the number of discoveries by discovery method and year**

# In[ ]:


df.groupby(['DiscoveryMethod', 'DiscoveryYear']).size().unstack().T.fillna(0).astype(int).sort_values(by=['DiscoveryYear'])


# ### Cuurently Transit method has become the most popular method for finding exoplanets

# ### RV 
# A star with a planet will move in its own small orbit in response to the planet's gravity. This leads to variations in the speed with which the star moves toward or away from Earth, i.e. the variations are in the radial velocity of the star with respect to Earth. The radial velocity can be deduced from the displacement in the parent star's spectral lines due to the Doppler effect.
# <img src="https://www.universetoday.com/wp-content/uploads/2017/12/Radial-Velocity-Method-star-orbits.png" style="height:500px">
# ### Direct Imaging
# (Usually in infrared)
# ### Microlensing
# Planets found due to gravitational lensing
# <img src="https://exoplanets.nasa.gov/system/resources/detail_files/53_hs-2012-07-b-web_print.jpg" style="height:500px">
# ### Timing
# Similar to transit method checking for time intervals between dips

# ### Checking for correlation in the data

# In[ ]:


numericCols=['PlanetaryMassJpt', 'RadiusJpt','PeriodDays', 'SemiMajorAxisAU', 'Eccentricity',
                 'SurfaceTempK','HostStarMassSlrMass', 'HostStarRadiusSlrRad',
                 'HostStarMetallicity','HostStarTempK','HostStarAgeGyr']
fig, ax = plt.subplots(figsize=(10,10))
ax=sns.heatmap(df[numericCols].corr(), annot=True, linewidths=.5,square=True)
print(ax.get_ylim())
ax.set_ylim(10.0, 0) #Due to few cropping issues in seaborn


# ### Interesting correlations
# - Surface Temp and Radius of planet
# - Host star mass and Radius of planet
# - Eccentricity and surface temperature
# 
# **Correlation does not imply causation!**
# 
# ### Obvious correlations
# - Host star mass and host star temp
# - Host star mass and host star radius
# - SurfaceTemp and Host star temp
# - Period and semi major axis
# 
# 
# 

# **Finding the percentage of null values in each column :**

# In[ ]:


df.isna().mean().sort_values(ascending=False).head(15)*100


# **We find AgeGyr, LongitudeDeg, AscendingNodeDeg, PeriastronDeg have more than 75% null values and hence won't be of much use**

# **Let's try finding the distribution of some columns like radius, mass, time period, semi major axis**

# In[ ]:


fig, axes = plt.subplots(2, 2,figsize=(18,12))
axes[0,0].hist(df["PlanetaryMassJpt"],bins=100)
axes[0,1].hist(df["RadiusJpt"],bins=100)
axes[1,0].hist(df["PeriodDays"],bins=100)
axes[1,1].hist(df["SemiMajorAxisAU"],bins=100)
plt.show()


# **As seen from histograms data is highly skewed**
# 
# 
# **If we attempt regression on such data it will fail miserably**

# **Attempting to plot a scatter of Mass vs Radius we find that the skew ruins the plot**

# In[ ]:


df=df.dropna(subset=['PlanetaryMassJpt', 'RadiusJpt'])#Remove null values
plt.scatter(df["RadiusJpt"],df["PlanetaryMassJpt"])


# #### Removing outliers to get a good scatter plot

# In[ ]:


df["RadiusJpt"].quantile(0.999)


# #### Removing outliers with values more than the 99.9th percentile we get an approximately polynomial graph between radius and mass as expected

# In[ ]:


dfremoved=df[df["RadiusJpt"]<df["RadiusJpt"].quantile(0.999)]
dfremoved=dfremoved[dfremoved["PlanetaryMassJpt"]<dfremoved["RadiusJpt"].quantile(0.999)]
plt.scatter(dfremoved["RadiusJpt"],dfremoved["PlanetaryMassJpt"])


# ### Log transformation
# #### Instead of trying to fit a polynomial we fill log transform the variables(Take their log values) and fit a straight line through them
# #### The slope of the line would give the degree of the polynomial

# In[ ]:


df["PlanetaryMassJptlog"]=df["PlanetaryMassJpt"].apply(np.log)
df["RadiusJptlog"]=df["RadiusJpt"].apply(np.log)

plt.scatter(df["RadiusJptlog"],df["PlanetaryMassJptlog"])


# ### Linear regression

# In[ ]:


linearRegressor = LinearRegression()


# In[ ]:


xTrain=df[["RadiusJptlog"]]
yTrain=df["PlanetaryMassJptlog"]


# In[ ]:


linearRegressor.fit(xTrain,yTrain)


# In[ ]:


plt.scatter(xTrain, yTrain, color = 'red')
plt.plot(xTrain, linearRegressor.predict(xTrain), color = 'blue')
plt.title('Radius vs Mass of planets')
plt.xlabel('log(Radius)')
plt.ylabel('log(Mass)')
plt.show()


# **Correlation coefficient**

# In[ ]:


linearRegressor.score(xTrain,yTrain)


# **Slope of line**

# In[ ]:


linearRegressor.coef_


# **Intercept**

# In[ ]:


linearRegressor.intercept_


# $log(Mass)=2.2*log(Radius)-0.45$
# 
# $Predicted : Mass=0.637*Radius^{2.2}$
# 
# 
# $Actual : Mass=c*Radius^3$

# ### Why is this incorrect?

# ## K-means Clustering

# **We will try clustering the planets into two groups based on mass and radius naturallywe expect them to be either rocky or gas giant**

# In[ ]:


km = KMeans(
    n_clusters=2, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)


# In[ ]:


X=np.array(df[["RadiusJptlog","PlanetaryMassJptlog"]])
X=X.reshape(-1,2)


# In[ ]:


km.fit(X)
y_km = km.fit_predict(X)


# In[ ]:


plt.scatter(
    X[y_km == 0, 0], X[y_km == 0, 1],
    s=50, c='lightgreen',
    edgecolor='black',
    label='gas giants'
)

plt.scatter(
    X[y_km == 1, 0], X[y_km == 1, 1],
    s=50, c='orange',
    edgecolor='black',
    label='rocky planets'
)
plt.title('Radius vs Mass of planets')
plt.xlabel('log(Radius)')
plt.ylabel('log(Mass)')

plt.legend(scatterpoints=1)
plt.show()


# ### TASK: Try checking if 2 is the best value for k (i.e are there only 2 clusters)

# In[ ]:


df["cluster"]=y_km


# In[ ]:


df["MassRadiusLogRatio"]=df["PlanetaryMassJptlog"]/df["RadiusJptlog"]


# In[ ]:


df.groupby("cluster").agg({"MassRadiusLogRatio":"mean"})


# In[ ]:


df["MassRadiusCubeRatio"]=df["PlanetaryMassJpt"]/df["RadiusJpt"]**3


# In[ ]:


df["MassRadiusCubeRatio"].hist()


# In[ ]:


df.groupby("cluster").agg({"MassRadiusCubeRatio":"mean"})


# In[ ]:


plt.boxplot([df[df["cluster"]==0]["MassRadiusCubeRatio"],df[df["cluster"]==1]["MassRadiusCubeRatio"]])
positions = (1, 2)
labels = ("Rocky planets", "Gas giants")
plt.ylabel('Density')
plt.xticks(positions, labels)
plt.show()


# In[ ]:


df=pd.read_csv("/kaggle/input/open-exoplanet-catalogue/oec.csv")


# ## Verifying keplers 3rd law

# ### Keplers 3rd Law
# $T^2\propto a^3$
# 
# where a is semi major axis

# **Plotting a scatter plot of period vs semi major axis**

# In[ ]:


plt.scatter(df["SemiMajorAxisAU"],df["PeriodDays"])


# In[ ]:


df=df.dropna(subset=['SemiMajorAxisAU', 'PeriodDays']) #Removing null values


# ### Doing log transformation and linear regression similar to before

# In[ ]:


df["SemiMajorAxisAUlog"]=df["SemiMajorAxisAU"].apply(np.log)
df["PeriodDayslog"]=df["PeriodDays"].apply(np.log)
plt.scatter(df["SemiMajorAxisAUlog"],df["PeriodDayslog"])


# In[ ]:


linearRegressor = LinearRegression()


# In[ ]:


xTrain=df[["SemiMajorAxisAUlog"]]
yTrain=df["PeriodDayslog"]


# In[ ]:


yTrain.shape


# In[ ]:


xTrain.describe()


# In[ ]:


yTrain.describe()


# In[ ]:


linearRegressor.fit(xTrain,yTrain)


# In[ ]:


plt.scatter(xTrain, yTrain, color = 'blue')
plt.plot(xTrain, linearRegressor.predict(xTrain), color = 'red',linewidth=5.0)
plt.title('Semi major axis vs time of orbit')
plt.xlabel('log(Semimajor axis)')
plt.ylabel('log(Time of orbit)')
plt.show()


# **Correlation coefficient**

# In[ ]:


linearRegressor.score(xTrain,yTrain)


# **Slope**
# 
# **(Expected 1.5)**

# In[ ]:


linearRegressor.coef_


# In[ ]:


linearRegressor.intercept_


# ## Again this is an incorrect analysis!
# ## The proportionality constant depends on mass of host star
# ## Hence Keplers law should be verified within a star system
# 

# In[ ]:





# In[ ]:


df.head()


# ### Making a linear regression function

# In[ ]:


def linreg(x,y):
    x=np.array(x).reshape(-1,1)
    y=np.array(y)
    linearRegressor = LinearRegression()
    linearRegressor.fit(x,y)
    return len(y),linearRegressor.score(x,y),linearRegressor.coef_[0],linearRegressor.intercept_
    


# ### Initializing variables for iterating

# In[ ]:


ra=""
dec=""
x=[]
y=[]
counts=[]
scores=[]
slopes=[]
intercepts=[]


# ### Planets from the same system occur together
# **So we'll do linear regression on the planets in the same star system and form a new dataframe with the results**

# In[ ]:


for row in df.itertuples(index=True, name='Pandas'):
    if(ra==getattr(row, "RightAscension") and dec==getattr(row, "Declination")):
        x.append(getattr(row, "SemiMajorAxisAUlog"))
        y.append(getattr(row, "PeriodDayslog"))
    else:
        if(len(x)>=2):
            count,score,slope,intercept=linreg(x,y)
            counts.append(count)
            scores.append(score)
            slopes.append(slope)
            intercepts.append(intercept)

        ra=getattr(row, "RightAscension")
        dec=getattr(row, "Declination")
        x=[getattr(row, "SemiMajorAxisAUlog")]
        y=[getattr(row, "PeriodDayslog")]
        
linreg_results=pd.DataFrame({"Count":counts,"Score":scores,"Slope":slopes,"Intercept":intercepts})  


# **This method is inefficient as it is not vectorised**
# ### TASK : Try vectorising the above code

# In[ ]:


linreg_results.head()


# In[ ]:


linreg_results.describe()


# **The standard deviation of slope is unnaturally high lets make a boxplot to find the problem**

# In[ ]:


plt.boxplot(linreg_results["Slope"])
plt.show()


# **We find one outlier having a very high slope so let's check after removing it**

# In[ ]:


linreg_results=linreg_results[linreg_results["Slope"]<6]
linreg_results.describe()


# **The standard deviation of slope is now reasonably good**

# In[ ]:


plt.hist(linreg_results["Slope"])
plt.show()


# # HR Diagram

# <img src="https://www.thoughtco.com/thmb/AAOKF09g0hT5BdaGf2xIK8Zz4b0=/768x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/HR_diagram_from_eso0728c-58d19c503df78c3c4f23f536.jpg" style="height:700px">

# In[ ]:


df["Luminosity"]=df["HostStarRadiusSlrRad"]**2*df["HostStarTempK"]**4
df["Luminositylog"]=np.log(df["Luminosity"])


# ### Plotting the HR Diagram we find a few outliers ruining the plot so we remove them
# ### The dot size in the scatterplot is mapped to the size of the star

# In[ ]:


cmap = matplotlib.colors.ListedColormap(["blue","blue","blue","blue","yellow","darkorange","red","red","red"][::-1])


# In[ ]:


fig = plt.figure(figsize=(5, 5))
plt.scatter(df["HostStarTempK"],df["Luminositylog"],c=df["HostStarTempK"],s=100*df["HostStarRadiusSlrRad"], cmap="coolwarm_r",edgecolor='black', linewidth=0.2)
plt.gca().invert_xaxis()
plt.title('HR Diagram')
plt.xlabel('HostStarTempK')
plt.ylabel('log(luminosity)')
plt.show()


# In[ ]:


df=df[df["HostStarTempK"]<25000]


# In[ ]:


fig = plt.figure(figsize=(18, 12))
points=plt.scatter(df["HostStarTempK"],df["Luminositylog"],c=df["HostStarTempK"],s=100*df["HostStarRadiusSlrRad"], cmap=cmap,edgecolor='black', linewidth=0.5)
plt.colorbar(points)
plt.gca().invert_xaxis()
plt.title('HR Diagram')
plt.xlabel('HostStarTempK')
plt.ylabel('log(luminosity)')
plt.show()


# ### As you can see the HR diagram provides a clean separation between the main sequence stars and giants

# ### TASK: Try clustering the HR diagram into the main sequence and giants

# ## Habitability

# #### A planet is habitable if it falls inside the habitable zone of the host star, where liquid water

# $\large inner radius(AU)=\sqrt{\frac{L}{1.1}}$
# 
# $\large outer radius(AU)=\sqrt{\frac{L}{0.53}}$
# 
# $\small L = Luminosity\:of\:host\:star$

# **If you want to know where these values came from check out this paper**
# 
# Kasting, James; Whitmire, Daniel; and Reynolds, Ray (1993). Habitable zones around main sequence stars. Icarus 101: 108-128.

# In[ ]:


df['Luminosity'] = df['HostStarRadiusSlrRad']**2  * (df['HostStarTempK']/5777)**4


# In[ ]:


#add habitable zone boundaries
df['HabZoneOut'] = np.sqrt(df['Luminosity']/0.53)
df['HabZoneIn'] = np.sqrt(df['Luminosity']/1.1)


# In[ ]:


habitable_zone=df[(df["SemiMajorAxisAU"]>df["HabZoneIn"]) & (df["SemiMajorAxisAU"]<df["HabZoneOut"])]
habitable_zone


# In[ ]:


habitable_zone=habitable_zone[(habitable_zone["PlanetaryMassJpt"]>0.0015) & (habitable_zone["PlanetaryMassJpt"]<0.03)]
habitable_zone


# In[ ]:


habitable_zone[["PlanetIdentifier","DistFromSunParsec"]].sort_values(by=["DistFromSunParsec"])


# ## TASK: Instead of hardcoding the parameters for habitability try finding the similarity of each exoplanet with earth and hence the most earth like exoplanets

# In[ ]:


df[df["ListsPlanetIsOn"]=="Solar System"]


# ### Task: Try clustering the planets in our solar system based on density ans verify if they form two clusters for rocky planets and gas giants

# In[ ]:




