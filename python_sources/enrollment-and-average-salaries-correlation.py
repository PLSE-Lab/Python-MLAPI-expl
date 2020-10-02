import pandas as pd
import numpy as np
import os

maschools = pd.read_csv("../input/MA_Public_Schools_2017.csv")
maschools.head()
#look at first five rows of maschools

list(maschools)
#display column names for maschools

enrollment = maschools['TOTAL_Enrollment']
#total enrollment value of each school

type(enrollment)
#enrollment is a pandas series

enrollment.head()
#first five rows of enrollment

enrollmentlist = pd.Series.tolist(enrollment)
#turn pandas series to list to work with it
type(enrollmentlist)

import matplotlib.pyplot as plt

plt.hist(enrollmentlist,bins=20)
#histogram of enrollment in Massachusetts schools

import numpy as np
print(np.mean(enrollmentlist))
#average school enrollment in MA

print(np.max(enrollmentlist))
#maximum school enrollment in MA

print(np.min(enrollmentlist))
#minimum school enrollment in MA

np.percentile(enrollmentlist,25)
#Q1/first quartile
 
np.percentile(enrollmentlist,75)
#Q3/third quartile

fil_enrollment = list(filter(lambda x: x!=0, enrollmentlist))
#remove values of zero

np.percentile(fil_enrollment,25)
#new first quartile value (Q1) without zero values

np.percentile(fil_enrollment,75)
#new third quartile value (Q3) without zero values

quartile1 = list(filter(lambda x: x<296.25, fil_enrollment))
#create list of values of enrollment in the first quartile

np.mean(fil_enrollment)
#mean without zero values

quartile2 = list(filter(lambda x: 296.25<x<515.54, fil_enrollment))
#list of enrollment values in second quartile

quartile3 = list(filter(lambda x: 515.54<x<618.75, fil_enrollment))
#list of enrollment values in third quartile

quartile4 = list(filter(lambda x: x>618.75, fil_enrollment))
#list of enrollment values in fourth quartile

plt.hist(quartile1)
#histogram of first quartile (lowest enrollment)

plt.hist(quartile2)
#histogram of second quartile (second lowest enrollment)

plt.hist(quartile3)
#histogram of third quartile (third lowest enrollment)

plt.hist(quartile4)
#histogram of fourth quartile (fourth lowest enrollment)

def printindex(idx):
#function that will return total enrollment and average salary provided an index number
    indenr =  maschools['TOTAL_Enrollment'][idx]
    indsal = maschools['Average Salary'][idx]
    print ('school number ' + str(idx) + ' has a total enrollment of ' + str(indenr) +' and an average salary of ' + str(indsal))

#create four empty lists of salary values
salq1 = []
salq2 = []
salq3 = []
salq4 = []

def takeindex(idx):
#fill in above four lists of salary values, each with average salaries of schools in each total enrollment quartile
#will the total enrollment of schools corrolate with their average salaries?
    indenr =  maschools['TOTAL_Enrollment'][idx]
    indsal = maschools['Average Salary'][idx]
    if indenr < 296.25:
        salq1.append(indsal)
    if 296.25 < indenr < 515.54:
        salq2.append(indsal)
    if 515.54 < indenr < 618.75:
        salq3.append(indsal)
    if 618.75 < indenr:
        salq4.append(indsal)

for x in range (0,1861):
    takeindex(x)
#apply above function to all rows of maschools

cleanedsalq1 = [x for x in salq1 if str(x) != 'nan']
cleanedsalq2 = [x for x in salq2 if str(x) != 'nan']
cleanedsalq3 = [x for x in salq3 if str(x) != 'nan']
cleanedsalq4 = [x for x in salq4 if str(x) != 'nan']
#remove nan values from all lists for cleaning

plt.hist(cleanedsalq1)
#average salaries of schools in the first total enrollment quartile

plt.hist(cleanedsalq2)
#average salaries of schools in the second total enrollment quartile

plt.hist(cleanedsalq3)
#average salaries of schools in the third total enrollment quartile

plt.hist(cleanedsalq4)
#average salaries of schools in the fourth total enrollment quartile
#it appears that schools with higher enrollment have higher average salaries - is this true?

import numpy as np
np.mean(cleanedsalq1)
#average salary for quartile 1 enrollment schools

np.mean(cleanedsalq2)
#average salary for quartile 2 enrollment schools

np.mean(cleanedsalq3)
#average salary for quartile 3 enrollment schools

np.mean(cleanedsalq4)
#average salary for quartile 4 enrollment schools

