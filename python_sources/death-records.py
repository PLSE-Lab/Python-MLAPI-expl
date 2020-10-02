# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy  # linear algebra
import pandas # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
print("Death python")

#importing the dataset into python


death = pandas.read_csv('../input/DeathRecords.csv')






#which gender died the most

genderdeath = death[:]['Sex']
genderlist = {}

for x in genderdeath:

	if x in genderlist:

		genderlist[x] = genderlist[x]+1

	else:

		genderlist[x] = 1

genderlist['Male'] = genderlist.pop('M')
genderlist['Female'] = genderlist.pop('F')
print(genderlist)




#marital deaths

maritaldeath = death[:]['MaritalStatus']
maritallist = {}

for x in maritaldeath:

	if x in maritallist:

		maritallist[x] = maritallist[x]+1

	else:

		maritallist[x] = 1


maritallist['Married'] = maritallist.pop('M')
maritallist['Single'] = maritallist.pop('S')
maritallist['Divorced'] = maritallist.pop('D')
maritallist['Unknown'] = maritallist.pop('U')
maritallist['Widowed'] = maritallist.pop('W')
print (maritallist)



#race deaths

racedeath = death[:]['Race']

racelist = {}




for x in racedeath:

	if x in racelist:

		racelist[x] = racelist[x]+1

	else:

		racelist[x] = 1




#print racelist




racelist['white'] = racelist.pop(1)

racelist['Black'] = racelist.pop(2)

racelist['American Indian'] = racelist.pop(3)

racelist['Chinese'] = racelist.pop(4)

racelist['Japanese'] = racelist.pop(5)

racelist['Hawaiian'] = racelist.pop(6)

racelist['Filipino'] = racelist.pop(7)

#racelist['Other Asian or Pacific Islander'] = racelist.pop(8)

racelist['Asian Indian'] = racelist.pop(18)

racelist['Korean'] = racelist.pop(28)

racelist['Samoan'] = racelist.pop(38)

racelist['Vietnamese'] = racelist.pop(48)

racelist['Guamanian'] = racelist.pop(58)

racelist['Other Asian or Pacific Islander'] = racelist.pop(68)

racelist['Combinder other asian or pacific islander'] = racelist.pop(78)





print (racelist)
#plt.bar(range(len(genderlist)), genderlist.values(), align='center')
#plt.xticks(range(len(genderlist)), list(genderlist.keys()))

#plt.show()

#manner of death
mannerdeath = death[:]['MannerOfDeath']
mannerlist = {}

for x in mannerdeath:

	if x in mannerlist:

		mannerlist[x] = mannerlist[x]+1

	else:

		mannerlist[x] = 1

mannerlist['Accident'] = mannerlist.pop(1)
mannerlist['Suicide'] = mannerlist.pop(2)
mannerlist['Homicide'] = mannerlist.pop(3)
mannerlist['Pending Investigation'] = mannerlist.pop(4)
mannerlist['Could not determine'] = mannerlist.pop(5)
#mannerlist['Self-Inflicted'] = mannerlist.pop(6)
mannerlist['Natural'] = mannerlist.pop(7)
mannerlist['Not Specified'] = mannerlist.pop(0)
print(mannerlist)

genedu = pandas.concat([death['Sex'], death['Education2003Revision']], axis=1)
malelist = {}
femalelist = {}
print(len(genedu))

for x in range(5):
    if genedu['Sex'][x] == 'M':
        if genedu['Education2003Revision'][x] in malelist:
            malelist[genedu['Education2003Revision'][x]] = malelist[genedu['Education2003Revision'][x]]  + 1
        else:
            malelist[genedu['Education2003Revision'][x]] =1
    if genedu['Sex'][x] == 'F':
        if genedu['Education2003Revision'][x] in femalelist:
            femalelist[genedu['Education2003Revision'][x]] = femalelist[genedu['Education2003Revision'][x]]  + 1
        else:
           femalelist[genedu['Education2003Revision'][x]] =1
       
print(malelist)
print(femalelist)


