#!/usr/bin/env python
# coding: utf-8

# **<font size=6>Basic analysis of the Google Play Store dataset</font>**

# ![imglink](https://cdn.freebiesupply.com/logos/large/2x/google-play-logo-png-transparent.png)

# 
# <br>
# **A Welcome message**<br>
# *Hi! This is my very first project here, needless to say that I am impressed with the plethora of the datasets and the wonderful kernels that you guys create! I am relativelly new in the Data Science, so I would sincerely appreciate if you share your opinion and perhaps correct any mistakes I might have along the way. I am ready to set sail!*
# <br><br>
# **<font size=5>Introduction</font>**
# <br>
# <font size=3>Creating a satisfying android application is nowadays something feasible for all, both programmers and non-programmers as well. This happens because developing apps in the Android Studio is one of the most discussed things online and you can find solutions for literally everything. Furthermore, for a non-programmer's case, there are plenty of programs to create applications for Android using simplified programming language, in a much more accessible way for the developer and there are even more programs which use a graphical environment, without the need of coding! </font>
# **<br><br><font size=3>Then, what does it take for an android app to be considered successful?</font><br><br>**
# <font size=3>First of all, what the word successful actually means? An Android application would be characterized as successful if it has an amount greater than a hundred thousand downloads. When the application is ready and uploaded on Play Store, advertisement is the primary factor for a successful application.</font>
# **<br><br><font size=3>But, what should we take into account before the application is completed?</font><br><br>**
#  <font size=3>Here is where a Data Scientist comes! Having a large amount of application data of Play Store can be very usefull to analyse why and how some applications succeed and others do not. So, the "Google Play Store App" dataset is a very good start to enable you to see things you may have to add or remove in your current developing Android application! Let's start!  </font>
# 
# 

# **<font size=5>Initialization</font>**

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set();
from sklearn.feature_extraction.text import CountVectorizer

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("../input/googleplaystore.csv")


# <font size=3>Take a glimpse in the dataset</font>

# In[ ]:


data.head()


# In[ ]:


data.count()


# **<font size=5>Data Cleaning</font>**

# <font size=3>First of, while I was looking the dataset, I realized something is going on in the 'Categories' column.</font>

# In[ ]:


data['Category'].unique()


# <font size=3>That '1.9 Category'  is a wrong value.  So I have to see in which rows '1.9' corresponds to.</font>

# In[ ]:


data[data['Category']=='1.9']


# <font size=3>The easy way is to just drop this row! But let's give it a try!</font>

# In[ ]:


data.loc[10472]


# <font size=3>Watching this row it seems that if we shift from 'Categories' and below, the values will make sense.</font>

# In[ ]:


data.loc[10472]=data.loc[10472].shift() # hole shift
#swap fisrt and second column
data['App'].loc[10472] = data['Category'].loc[10472]
data['Category'].loc[10472] = np.nan
data.loc[10472]


# <font size=3>Now it's time to clean the dataset from indentical records.</font>

# In[ ]:


data[data.duplicated()].count()


# In[ ]:


data=data.drop_duplicates()
data[data.duplicated()].count()


# <font size=3>After this point, we will convert the numeric columns into integers or floats accordingly.</font>

# In[ ]:


data.dtypes


# <font size=3>We can visually see that all the data is object type. The desirable columns to convert are: </font>**<font size=3>Rating, </font>** **<font size=3>Reviews, </font>** **<font size=3>Size, </font>** **<font size=3>Installs, </font>** <font size=3> and </font> **<font size=3>Price.</font>** <font size=3> I will not conclude the </font> **<font size=3>Last Updated, </font>** **<font size=3>Current Ver</font>** <font size=3> and </font> **<font size=3>Android Ver </font>** <font size=3>in this project   :( </font>

# <font size=4>&#8195; &#8195;  &#8195; &#8195; &#8194; 1. Rating</font>

# In[ ]:


data['Rating'].unique()


# In[ ]:


data['Rating'] = pd.to_numeric(data['Rating'],errors='coerce')


# <font size=4>&#8195; &#8195;  &#8195; &#8195; &#8194; 2. Reviews</font>

# In[ ]:


data['Reviews'].unique()


# In[ ]:


data['Reviews'] = pd.to_numeric(data['Reviews'],errors='coerce')


# <font size=4>&#8195; &#8195;  &#8195; &#8195; &#8194; 3. Size</font>

# In[ ]:


data['Size'].unique()


# <font size=3>The sizes will be converted into Kilobytes.  Those of which are equal to 'Varies with device' will be turned into 'NaN', because it may cause  false fluctuations in Visualization. Let's be happy with what we have!</font>

# In[ ]:


data['Size'].replace('Varies with device', np.nan, inplace = True ) 
data['Size']=data['Size'].str.extract(r'([\d\.]+)', expand=False).astype(float) *     data['Size'].str.extract(r'([kM]+)', expand=False).fillna(1).replace(['k','M'],[1,1000]).astype(int)


# <font size=4>&#8195; &#8195;  &#8195; &#8195; &#8194; 4. Installs</font>

# In[ ]:


data['Installs'].unique()


# <font size=3> We will just ignore '+' symbol and remove commas.</font>

# In[ ]:


data['Installs']=data['Installs'].str.replace(r'\D','').astype(float)


# <font size=4> &#8195; &#8195;  &#8195; &#8195; &#8194; 5. Price</font>

# In[ ]:


data['Price'].unique()


# In[ ]:


data['Price']=data['Price'].str.replace('$','').astype(float)


# **<font size=5>Visualization</font>**

# <font size=3>As I already mentioned, the more downloads (Installations) a mobile application has, the more successful it is considered. Therefore, this project focuses on the Installations and how other factors can affect them. </font> 
# <br>**<font size=3> My goal is to make an app with more that 100,000 downloads.</font>**

# <font size=4>&#8195; &#8195;  &#8195; &#8195; &#8194; 1. Category</font>

# <font size=3>To begin with, let's see how many applications we have in each Category.</font>

# In[ ]:


plt.figure(figsize=(10,10))
g = sns.countplot(y="Category",data=data, palette = "Set2")
plt.title('Total apps of each Category',size = 20)


# <font size=3>We can see that Family, Game and Tools are the most frequent Categories in our dataset. But what happens with the Installs in each Category?</font>

# In[ ]:


plt.figure(figsize=(10,10))
g = sns.barplot(x="Installs", y="Category", data=data, capsize=.6)
plt.title('Installations in each Category',size = 20)


# <font size=3>Of course! Communication and Social Categories have the most downloads! Let's see which are the Categories of our interest.</font>

# In[ ]:


data[data[['Installs']].mean(axis=1)>1e5]['Category'].unique()


# <font size=4>&#8195; &#8195;  &#8195; &#8195; &#8194; 2. Rating</font>

# In[ ]:


plt.figure(figsize=(10,10))
plt.scatter( x=data['Rating'], y=data['Installs'] , color = 'blue')
g = sns.lineplot(x="Rating", y="Installs",color="red", data=data) 
plt.yscale('log')
plt.xlabel('Rating')
plt.ylabel('Installs')
plt.title('Rating-Installs (Scatter & line plot)',size = 20)
plt.show()
    


# <font size=3>So it seems that a Rating between 2.8 and 4.8 is desirable. But over 4.8 the Installations drop significantly and that makes sense. Imagine a Youtube video with millions of views. There is no way to have 100% likes. But a video with only a few views (as many as youtuber's friends and relatives :P ), is quite possible to have zero dislikes. That is what exactly happens with such high Ratings!</font>

# <font size=4>&#8195; &#8195;  &#8195; &#8195; &#8194; 3. Reviews</font>

# In[ ]:


g = sns.lmplot(y="Installs",x="Reviews", data=data,size=(10))
plt.xscale('log')
plt.yscale('log')
plt.title('Reviews-Installs ',size = 20)


# <font size=3>Here is the correlation we are searching for! The Reviews, unlike Rating, describe the Installs in a linear way. </font>

# <font size=4>&#8195; &#8195;  &#8195; &#8195; &#8194; 4. Size</font>

# In[ ]:


plt.figure(figsize=(10,10))
g = sns.boxplot(x="Installs", y="Size", data=data)
g.set_xticklabels(g.get_xticklabels(), rotation=40, ha="right")
plt.title('Installs-Size(kilobyte) ',size = 20)


# <font size=3>After viewing this graph, we can conclude that the more Installs are increase, the Size increases as well. But why is that? One of the most appealing things in an app is its appearance. Realistic pictures, 3D models, dazzling animations and beautiful music make up for a more pleasant experience to the user. But all of the above seriously increase the size of the app . So this graph basically says:</font> *<font size=3> Your app has to be beautiful!</font>*

# <font size=4>&#8195; &#8195;  &#8195; &#8195; &#8194; 5. Content Rating</font>

# In[ ]:


plt.figure(figsize=(10,10))
ax = sns.barplot(y="Installs", x="Content Rating", data=data, capsize=.5)
ax.set_xticklabels(ax.get_xticklabels(),  ha="left")
plt.title('Content Rating-Installs',size = 20)

##pie plot
labels=data['Content Rating'].unique()
explode = (0.1, 0, 0, 0, 0)
size=list()
for content in labels:
    size.append(data[data['Content Rating']==content]['Installs'].mean())

##merging Unrated & Adults 
labels[4] = 'Unrated &\n Adults only 18+'
labels = np.delete(labels,5)
size[4]=size[4]+size[5]
size.pop()

plt.figure(figsize=(10,10))
colors = ['#ff6666','#66b3ff','#99ff99','#ffcc99', '#df80ff']
plt.pie(size, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.axis('equal')
plt.title('Percentage of Installs for each Content Rating',size = 20)
plt.show()


# <font size=3>It is obvious that the applications which are permissible to children, have the most Downloads.</font>

# <font size=4>&#8195; &#8195;  &#8195; &#8195; &#8194;  6. Type and Price</font>

# In[ ]:


plt.figure(figsize=(10,10))

labels=['Apps with less than 100,000 Downloads', 'Apps with more than 100,000 Downloads']
size=list()
size.append(data['App'][data['Installs']<1e5].count()) 
size.append(data['App'][data['Installs']>=1e5].count()) 

labels_inner=['Free', 'Paid', 'Free', 'Paid']
size_inner=list()
size_inner.append(data['Type'][data['Type']=='Free'][data['Installs']<1e5].count()) 
size_inner.append(data['Type'][data['Type']=='Paid'][data['Installs']<1e5].count()) 
size_inner.append(data['Type'][data['Type']=='Free'][data['Installs']>=1e5].count())
size_inner.append(data['Type'][data['Type']=='Paid'][data['Installs']>=1e5].count()) 

colors = ['#99ff99', '#66b3ff']
colors_inner = ['#c2c2f0','#ffb3e6', '#c2c2f0','#ffb3e6']

explode = (0,0) 
explode_inner = (0.1,0.1,0.1,0.1)

#outer pie
plt.pie(size,explode=explode,labels=labels, radius=3, colors=colors)
#inner pie
plt.pie(size_inner,explode=explode_inner,labels=labels_inner, radius=2, colors=colors_inner)
       
#Draw circle
centre_circle = plt.Circle((0,0),1.5,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.axis('equal')
plt.tight_layout()
plt.show()


# <font size=3>This figure, at first, shows that the number of apps with more than 100,000 Downloads is about the same as the number of apps with less. After that we cannot bring into any conclusion on how the Type affects the number of Installs because Paid apps are much more less in number than the Free apps. Thus, I would choose to make a Free application like the majority does!</font> 

# In[ ]:


g = sns.lineplot(x="Price", y="Installs", data=data)

plt.figure(figsize=(10,10))
g = sns.lineplot(x="Price", y="Installs", data=data)
g.set(xlim=(0, 10))
plt.title('Price (0-10$) - Installs',size = 20)


# <font size=3>But, if you are planning on charging it, just sell it below 1 dolar!  </font>

# <font size=4>&#8195; &#8195;  &#8195; &#8195; &#8194;  7. Name</font>

# In[ ]:


corpus=list(data['App'])
vectorizer = CountVectorizer(max_features=50, stop_words='english')
X = vectorizer.fit_transform(corpus)
names=vectorizer.get_feature_names()
values=X.toarray().mean(axis=0)

plt.figure(figsize=(15,15))
sns.barplot(x=values, y=names, palette="viridis")
plt.title('Top 50 most frequently occuring words',size = 20)


# **<font size=4> So we got to the end! Just name it and start developing!!!</font>**

# 
