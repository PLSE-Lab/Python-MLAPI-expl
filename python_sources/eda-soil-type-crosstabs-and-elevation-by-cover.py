#!/usr/bin/env python
# coding: utf-8

# Extending the really good EDA by Anika (and inspired by Fatih Bilgin), I offer some work cross-tabbing soil type, and showing some elevations by cover and soil type.
# 
# Please tell me my style failures and code awkwardness, this is my first notebook and I'm here to learn.
# 
# tldr;
# - Anika shows that elevation holds a lot of information about cover type
# - Cross-tabbing soil type shows it too holds much information
# - Plotting elevation by soil type and cover type suggests soil type may help discriminate among covers

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


train = pd.read_csv('/kaggle/input/learn-together/train.csv')
train.head()


# In[ ]:


def label_soil(row):
    ''' flatten boolean soil types to a single column'''
    for ndx in range(1, 40):
        label = 'Soil_Type{}'.format(ndx)
        if row[label] == 1:
            return label


# In[ ]:


# copied from Anika, https://www.kaggle.com/sinaasappel/trees-in-roosevelt-national-forest-eda

def CoverType(row):
    if row.Cover_Type == 1:
        return 'Spruce/Fir'
    elif row.Cover_Type == 2:
        return 'Lodgepole Pine'
    elif row.Cover_Type == 3:
        return 'Ponderosa Pine'
    elif row.Cover_Type == 4:
        return 'Cottonwood/Willow'
    elif row.Cover_Type == 5:
        return 'Aspen'
    elif row.Cover_Type == 6:
        return 'Douglas-fir'
    else:
        return 'Krummholz'


# In[ ]:


train['SoilType'] = train.apply(lambda x: label_soil(x), axis=1)


# In[ ]:


train['CoverType'] = train.apply(CoverType, axis='columns')


# Cross-tabulate soil and cover type, sorting output by row totals

# In[ ]:


soil_cross = pd.crosstab(train.SoilType, train.CoverType)
# total of row
soil_cross.loc[:,'Total']= soil_cross.sum(axis=1)
# total of column
soil_cross.loc['Total', :]= soil_cross.sum(axis=0)


soil_cross.sort_values('Total', ascending=False)


# Convert data description of soil type to dictionary . . . 

# In[ ]:


soil_types = {
    'Soil_Type1': 'Cathedral family - Rock outcrop complex, extremely stony.',
    'Soil_Type2': 'Vanet - Ratake families complex, very stony.',
    'Soil_Type3': 'Haploborolis - Rock outcrop complex, rubbly.',
    'Soil_Type4': 'Ratake family - Rock outcrop complex, rubbly.',
    'Soil_Type5': 'Vanet family - Rock outcrop complex complex, rubbly.',
    'Soil_Type6': 'Vanet - Wetmore families - Rock outcrop complex, stony.',
    'Soil_Type7': 'Gothic family.',
    'Soil_Type8': 'Supervisor - Limber families complex.',
    'Soil_Type9': 'Troutville family, very stony.',
    'Soil_Type10': 'Bullwark - Catamount families - Rock outcrop complex, rubbly.',
    'Soil_Type11': 'Bullwark - Catamount families - Rock land complex, rubbly.',
    'Soil_Type12': 'Legault family - Rock land complex, stony.',
    'Soil_Type13': 'Catamount family - Rock land - Bullwark family complex, rubbly.',
    'Soil_Type14': 'Pachic Argiborolis - Aquolis complex.',
    'Soil_Type15': 'unspecified in the USFS Soil and ELU Survey.',
    'Soil_Type16': 'Cryaquolis - Cryoborolis complex.',
    'Soil_Type17': 'Gateview family - Cryaquolis complex.',
    'Soil_Type18': 'Rogert family, very stony.',
    'Soil_Type19': 'Typic Cryaquolis - Borohemists complex.',
    'Soil_Type20': 'Typic Cryaquepts - Typic Cryaquolls complex.',
    'Soil_Type21': 'Typic Cryaquolls - Leighcan family, till substratum complex.',
    'Soil_Type22': 'Leighcan family, tibbll substratum, extremely bouldery.',
    'Soil_Type23': 'Leighcan family, till substratum - Typic Cryaquolls complex.',
    'Soil_Type24': 'Leighcan family, extremely stony.',
    'Soil_Type25': 'Leighcan family, warm, extremely stony.',
    'Soil_Type26': 'Granile - Catamount families complex, very stony.',
    'Soil_Type27': 'Leighcan family, warm - Rock outcrop complex, extremely stony.',
    'Soil_Type28': 'Leighcan family - Rock outcrop complex, extremely stony.',
    'Soil_Type29': 'Como - Legault families complex, extremely stony.',
    'Soil_Type30': 'Como family - Rock land - Legault family complex, extremely stony.',
    'Soil_Type31': 'Leighcan - Catamount families complex, extremely stony.',
    'Soil_Type32': 'Catamount family - Rock outcrop - Leighcan family complex, extremely stony.',
    'Soil_Type33': 'Leighcan - Catamount families - Rock outcrop complex, extremely stony.',
    'Soil_Type34': 'Cryorthents - Rock land complex, extremely stony.',
    'Soil_Type35': 'Cryumbrepts - Rock outcrop - Cryaquepts complex.',
    'Soil_Type36': 'Bross family - Rock land - Cryumbrepts complex, extremely stony.',
    'Soil_Type37': 'Rock outcrop - Cryumbrepts - Cryorthents complex, extremely stony.',
    'Soil_Type38': 'Leighcan - Moran families - Cryaquolls complex, extremely stony.',
    'Soil_Type39': 'Moran family - Cryorthents - Leighcan family complex, extremely stony.',
    'Soil_Type40': 'Moran family - Cryorthents - Rock land complex, extremely stony.',
}


# . . . use dictionary to give geologic information alongside cross-tabs

# In[ ]:


soil_cross['st'] = soil_cross.index 
soil_cross['description'] = soil_cross.apply(lambda x: soil_types.get(x['st']), axis=1)
soil_cross.sort_values('Total', ascending=False)


# In[ ]:


# copied from Anika (link above)

# make 7 new train sets, one for each cover type
spruce = train[train.Cover_Type == 1]
lodgepole = train[train.Cover_Type == 2]
ponderosa = train[train.Cover_Type == 3]
cottonwood = train[train.Cover_Type == 4]
aspen = train[train.Cover_Type == 5]
douglas = train[train.Cover_Type == 6]
krummholz = train[train.Cover_Type == 7]

# stash the results under labels for later iteration:
covers = {
    'spruce': spruce,
    'lodgepole': lodgepole, 
    'ponderosa': ponderosa, 
    'cottonwood': cottonwood, 
    'aspen': aspen, 
    'douglas': douglas, 
    'krummholz': krummholz,
}


# In[ ]:


# copied from Anika, https://www.kaggle.com/sinaasappel/trees-in-roosevelt-national-forest-eda

sns.distplot(a = spruce['Elevation'], label = "Spruce")
sns.distplot(a = lodgepole['Elevation'], label = "Lodgepole")
sns.distplot(a = ponderosa['Elevation'], label = "Ponderosa")
sns.distplot(a = cottonwood['Elevation'], label = "Cottonwood")
sns.distplot(a = aspen['Elevation'], label = "Aspen")
sns.distplot(a = douglas['Elevation'], label = "Douglas")
sns.distplot(a = krummholz['Elevation'], label = "Krummholz")

# Add title
plt.title("Histogram of Elevation, by cover type")

# Force legend to appear
plt.legend()


# In[ ]:


def plot_soil_type(st_num):
    ''' plot elevation by soil type'''
    this_df = train[train.SoilType == 'Soil_Type{}'.format(st_num)]
    sns.distplot(a = this_df['Elevation'], label = "Type{}".format(st_num))

plot_soil_type('10')
plot_soil_type('29')
plot_soil_type('3')
plot_soil_type('4')
plot_soil_type('23')
plot_soil_type('38')
plot_soil_type('30')
    
plt.title("Histogram of Elevation, by soil type")
plt.legend()


# The first seven soil types also group by elevation -- more analysis needed to see if this proxies cover type elevation, or helps discriminate

# In[ ]:


def cover_by_soil(st_num):
    ''' plot elevation by soil type and cover'''
    for label, df in covers.items():
        this_df = df[df.SoilType == 'Soil_Type{}'.format(st_num)]
        sns.distplot(a = this_df['Elevation'], label=label)
        plt.title('Elevation, by cover type -- Soil Type {}'.format(st_num))
        plt.legend()

cover_by_soil('10')


# Soil Type 10 looks a lot like the overall elevation histogram.  But, spikes like the following may suggest some help distinguishing covers within an elevation.

# In[ ]:


cover_by_soil(3)


# In[ ]:


cover_by_soil(23)


# In[ ]:


Thank you for considering these ideas.  Suggestions for improvement are most welcome.


# 
