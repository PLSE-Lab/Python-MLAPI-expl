#!/usr/bin/env python
# coding: utf-8

# The Data-Set contains offerings of used cars in germany (20 attributes, 371528 examples). These cars where submitted to the website 'ebay Kleinanzeigen' and have been crawled between 2016-03-05 and 2016-04-07. In the following we are going to take a look at the data, clean it step by step and reach the following two sets:
# 
#  - 'cleaned_data': containing 312.729 examples, 15 attributes and some
#    missing values
#  - 'superclean_data': containing 247.872 examples, 15 attributes and not
#    a single missing value
# 
# Thanks to 'Orges Leka' for providing this Data-Set.
# 
# Feel free to comment on anything you find interesting or unnecessary so I can learn from my first submitted Kernel.
# Have fun!
# 
# Morris
# 
# 
# We can use these variables to evaluate the set:
# 
#     dateCrawled         : when advert was first crawled, all field-values are taken from this date
#     name                : headline, which the owner of the car gave to the advert
#     seller              : 'privat'(ger)/'private'(en) or 'gewerblich'(ger)/'dealer'(en)
#     offerType           : 'Angebot'(ger)/'offer'(en) or 'Gesuch'(ger)/'request'(en)
#     price               : the price on the advert to sell the car
#     
#     abtest              : ebay-intern variable (argumentation in discussion-section)
#     vehicleType         : one of eight vehicle-categories 
#     yearOfRegistration  : at which year the car was first registered
#     gearbox             : 'manuell'(ger)/'manual'(en) or 'automatik'(ger)/'automatic'(en)
#     powerPS             : the power of the car in PS
#     
#     model               : the cars model
#     kilometer           : how many kilometres the car has driven
#     monthOfRegistration : at which month the car was first registered
#     fuelType            : one of seven fuel-categories
#     brand               : the cars brand
#     
#     notRepairedDamage   : if the car has a damage which is not repaired yet
#     dateCreated         : the date for which the advert at 'ebay Kleinanzeigen' was created
#     nrOfPictures        : number of pictures in the advert
#     postalCode          : where in germany the car is located
#     lastSeenOnline      : when the crawler saw this advert last online
#     
# Let's have a first peek.

# In[ ]:


#Import
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

#Data loading
path=r'../input/autos.csv'
all_data = pd.read_csv(path, sep=',',encoding='Latin1')

all_data.describe()


# In[ ]:


all_data.head(3)


# There are quite a few strange values and columns. 
# For cleaning purposes I'm going to inspect each column for itself and will remove undesired rows to counteract the data-distortion.

# In[ ]:


#inspecting: nrOfPictures
print(all_data['nrOfPictures'].sum(), 'pictures in all offers combined')


# In[ ]:


#there are no pictures at all in these adverts: bye bye column!
work_data=all_data.drop('nrOfPictures',1)


#inspecting: seller
print(work_data.groupby('seller').size())


# In[ ]:


#we can dismiss the three rows containing 'gewerblich' and then get rid of the column 'seller'. 
#All adverts are now submitted by private sellers.
work_data = work_data[work_data.seller != 'gewerblich']
work_data=work_data.drop('seller',1)


#inspecting: offerType
print(work_data.groupby('offerType').size())


# In[ ]:


#Deja vu. We won't need those twelve people asking for a car in our data.
work_data = work_data[work_data.offerType != 'Gesuch']
work_data=work_data.drop('offerType',1)


#inspecting: name
print(len(work_data.groupby('name').size()), 'different names for all offers')


# In[ ]:


#So many unique names for all these adverts. Grouping them for interesting conclusions will be tough.
#Perhaps someone of you can distil some useful data out of the strings. 
#I can't yet, so we'll leave the column behind.
work_data=work_data.drop('name',1)

#The column 'abtest' has been hard to interpret. After consulting the original database uploader, 
#we concluded to this interpretation:
#It contains an ebay-intern variable not of interest. 
#You can monitor the conversation and argumentation in the discussion section.
work_data=work_data.drop('abtest',1)


# In[ ]:


#The remaining columns look useful and we're diving to the next level.
#We'll need a little sample of our data to save some time plotting everything. 
sample_data=work_data.sample(n=10000, random_state=1)



#inspecting: price
plt.subplot(3,1,1)
sample_data['price'].hist(bins=50)
plt.title('Original-Histogram price')
plt.show()
#There are kinda expensive cars on sale. The owners are slightly overestimating the value I guess.

#I'll' cut the price at 100000
work_data = work_data[work_data.price < 100000]
sample_data=work_data.sample(n=10000, random_state=1)

plt.subplot(3,1,2)
sample_data['price'].hist(bins=50)
plt.title('Cut-Histogram price')
plt.show()



print(len(work_data[work_data.price == 0]), 'cars with price 0')
#The other way round some people are giving their car away for free. Unlikly in such high rates. Remove!
work_data = work_data[work_data.price != 0]
sample_data=work_data.sample(n=10000, random_state=1)

plt.subplot(3,1,3)
sample_data['price'].hist(bins=50)
plt.title('Final-Histogram price')
plt.show()


# Price-distribution looks quite nice now. Our set contains every reasonably priced car offered by a private person at this point.

# In[ ]:


#Assuming someone would really sell the first motorized car ever build (1863), 
#we can cut of every advert proclaiming selling older ones.
#Used cars from 2017 and even younger time travelling cars are unlikly as well.
work_data = work_data[(work_data.yearOfRegistration >= 1863) & (work_data.yearOfRegistration < 2017)]


#inspecting: powerPS
print(work_data['powerPS'].describe())


# In[ ]:


#We'll remove the so-called cars with 0 PS.
#More than 1000 PS are suspicious too. 
work_data = work_data[(work_data.powerPS > 0) & (work_data.powerPS < 1000)]


# In[ ]:


print(work_data['powerPS'].describe())


# Almost ready. Lets check the 'cleaned_data'.
# Remember there are quite a few missing values in this 312.729 data-rows. If you want to inspect it yourself, keep the missing values in the back of your head interpreting your results.
# 
# I'll provide 'superclean_data' if you are okay with fewer examples to work with. 

# In[ ]:


#Your playground is ready, have fun!
cleaned_data = work_data

#If you inspect the columns with digits (price, yearOfRegistration, powerPS, kilometer, 
#monthOfRegistration, postalcode) with this little code, you can see the data is now well behaved.
#Write the column name you want to inspect into the string-quotes 
column_name_you_want_to_inspect='yearOfRegistration'
print(cleaned_data[column_name_you_want_to_inspect].describe())

sample_data=cleaned_data.sample(n=10000, random_state=1)
sample_data[column_name_you_want_to_inspect].hist(bins=20)
plt.title('Cleaned-Histogram ' + column_name_you_want_to_inspect)
plt.show()


# In[ ]:


#clean, let's make it superclean by removing all rows with missing values
superclean_data = cleaned_data.dropna()
print(superclean_data.describe())


# In[ ]:


superclean_data.to_csv('superclean_data.csv', index=False)


# In[ ]:


cleaned_data.to_csv('clean_data.csv', index=False)


# I think we have reached our final destination. You have now two cleaned datasets to work with.
# 
# A quick recap what is left:
# 
# You can explore used cars, which were offered on 'ebay Kleinanzeigen' in germany. All remaining examples were submitted by private citizens. Useless columns and outliers have been removed.
# 
#  1. 'cleaned_data' contains 312.729 examples, 15 attributes and some
#     missing values
#  2. 'superclean_data' contains 247.872 examples, 15 attributes and not a
#     single missing value
# 
# I'd be glad, if you point out some room for improvement.
# Thanks for participating!
# 
# Morris
