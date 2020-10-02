#!/usr/bin/env python
# coding: utf-8

# # Part I: Departmental and Policy Data Analysis

# Goal: Calculate correlation between numPolicies and numkillings and calculate correlation per policy with numkillings.
# 
# Step 1: Read and clean data. This is using US census data from mappingpoliceviolence.com and policy data from Campaign Zero. The MPV data was already in a CSV, but the CZ dataset wasn't. This required converting it using Optical Character Recognition Software, and then formatting that into the same CSV file as the MPV data. After that, I put all the column names and row names into a dictionary to make queries easier.

# In[ ]:


import csv
import numpy as np

#read CSV
with open('policydept.csv') as csvfile:
    csvthing = csv.reader(csvfile, delimiter=',')
    total=0
    count=0
    readCSV=[]
    for row in csvthing:
        readCSV.append(row)

    #dictify row 1
    querydict={}
    labels=readCSV[0]
    for el in labels:
        name=el
        querydict[name]=count
        count=count+1
    count=0

    #dictify column 1
    deptdict={}
    for row in readCSV:
        if row[0]!='City':
            dept=row[0]
            deptdict[dept]=count
            count=count+1
    count=0
    
    print(deptdict)
    print(querydict)


# ## Correlation with Number of Policies

# Getting data from the CSV file about policies, police homicide rates, and disparities.

# In[ ]:


nps=[]
rates=[]
ratebs=[]
disps=[]
for dept in deptdict.keys():
    print('dept is', dept)
    i=deptdict[dept]+1
    j=querydict['numPolicies']
    nP=int(readCSV[i][j])
    nps.append(nP)
    print('numPolicies is', nP)
    j=querydict['killingsPerMillion']
    rate=float(readCSV[i][j])
    rates.append(rate)
    rateb=float(readCSV[i][querydict['Avg Annual Police Homicide Rate for Black People']])
    ratebs.append(rateb)
    print('police homicide rate is', rate)
    disp=float(readCSV[i][querydict['Black-White Disparity']])+float(readCSV[i][querydict['Hispanic-White Disparity']])
    disps.append(disp)
    print('racial disparity is ',disp)
    print()


# As shown below there isn't the greatest negative correlation between number of policies and rate of killings. This could be due to varied crime rates and implementation of policies. I decided it might be better to look at individual policies

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('fivethirtyeight')

plt.scatter(nps,ratebs)
print(np.corrcoef(nps,ratebs)[0][1])


# Looking at individual policies still didn't show the most correlation either. This made me realize that there was an even bigger problem.

# In[ ]:


for j in range(32,41):
    policys=[]
    print(readCSV[0][j])
    for dept in deptdict.keys():
        i=deptdict[dept]+1
        nP=int(readCSV[i][j])
        policys.append(nP)
    plt.scatter(policys,ratebs)
    print('correlation:',np.corrcoef(policys,ratebs)[0][1])


# I then restricted to similar cities to avoid noise due to differing factors. I thought it would be ideal to compare a city to itself before the policy but I didn't have that data so I had to use similar cities. As shown below, this effectively revealed that policies really did correlate to lower rates of police homicide.

# In[ ]:


def lookAtCities(bigcities,s):
    ratesrestricted=[]
    for dept in bigcities:
        i=deptdict[dept]+1
        query='Avg Annual Police Homicide Rate'+s
        j=querydict[query]
        rate=float(readCSV[i][j])
        ratesrestricted.append(rate)
    for j in range(32,40):
        policys=[]
        print(readCSV[0][j])
        for dept in bigcities:
            i=deptdict[dept]+1
            nP=int(readCSV[i][j])
            policys.append(nP)
        print('correlation:',np.corrcoef(policys,ratesrestricted)[0][1])

#comparing similar cities yield higher negative correlation for most policies
print('Looking at rate of black homicides')
lookAtCities(['Los Angeles','San Francisco','Philadelphia','Orlando'],' for Black People')
lookAtCities(['New Orleans','Baton Rouge','Orlando'],' for Black People')
print()
print('Looking at rate of white homicides')
lookAtCities(['Los Angeles','San Francisco','Philadelphia','Orlando'],'')
lookAtCities(['New Orleans','Baton Rouge','Orlando'],'')


# Finally, I thought that maybe bucketing homicide rates based on number of policies might reveal some larger trend since I saw a graphic indicating so published by Campaign Zero, who provided the policies dataset. After doing this I realized they slightly painted the picture brighter than it was by drawing the below graph in a distorted way. It probably would have been better for them to compare similar cities to get the trend they were looking for.

# In[ ]:


import numpy as np
labels=['0 to 1','2','3','4+']
labelsn=[0,1,2,3]
lowest=0
ok=0
better=0
best=0
for row in readCSV:
    if row[0]!='City':
        nP=int(row[40])
        k=float(row[42])
        if nP<2:
            lowest=lowest+k
        elif nP==2:
            ok=ok+k
        elif nP==3:
            better=better+k
        elif nP>4:
            best=best+k
killings=[lowest,ok,better,best]
plt.plot(labels,killings)
print('correlation:',np.corrcoef(labelsn,killings)[0][1])


# # Part II: State Specific Data Analysis

# In[ ]:


import csv
#can use rate of death estimates to estimate p-values?

#read csv
with open('states.csv') as csvfile:
    csvthing = csv.reader(csvfile, delimiter=',')
    total=0
    count=0
    readCSV=[]
    for row in csvthing:
        readCSV.append(row)

    #dictify row 1
    querydict={}
    labels=readCSV[0]
    for el in labels:
        name=el
        querydict[name]=count
        count=count+1
    count=0

    #dictify column 1
    statedict={}
    for row in readCSV:
        if row[0]!='State':
            state=row[0]
            statedict[state]=count
            count=count+1
    count=0
    
    print(statedict)
    print(querydict)


# ## Ex.1 Alabama

# We can look at the state of Alabama as an example before algorithmically analyzing all of the states. I used bayesian probability to derive how to calculate the rate of police homicide given race in a given state. The derivation of this (and thus all following formulas) is shown on my github using Bayes theoerem. Then using this probability and assuming a Poisson distribution(a probability distribution where something happens at a rate) I calculated the rate of homicide. This revealed significant racial disparities. Also if these numbers sound very large, it's important to remember these homicides include armed victims.

# In[ ]:


pb1=float(readCSV[statedict['Alabama']+1][6])/float(readCSV[statedict['Alabama']+1][2].replace(',',''))
print('the probability that someone is killed by an officer given theyre black is',pb1)
#calculating lambda=p*n
print('the rate of killing per 100k is thus',(pb1*100000))
rate1=pb1*100000

whitepop=float(readCSV[statedict['Alabama']+1][17])*int(readCSV[statedict['Alabama']+1][1].replace(',', ''))
pb2=float(readCSV[statedict['Alabama']+1][11])/(whitepop)
print('the probability that someone is killed by an officer given theyre white is',pb2)
#calculating lambda=p*n
print('the rate of killing per 100k is thus',pb2*100000)
rate2=pb2*100000

pb3=(float(readCSV[statedict['Alabama']+1][6])+float(readCSV[statedict['Alabama']+1][7])
    +float(readCSV[statedict['Alabama']+1][8])+float(readCSV[statedict['Alabama']+1][9])
    +float(readCSV[statedict['Alabama']+1][10]))/(int(readCSV[statedict['Alabama']+1][1].replace(',', ''))-whitepop)
print('the probability that someone is killed by an officer given theyre not white is',pb3)
#calculating lambda=p*n
print('the rate of killing per 100k is thus',pb3*100000)
rate3=pb3*100000

bwdisparity =rate1-rate2
print('the black-white disparity is ',bwdisparity)

disparity=rate3-rate2
print('the disparity is ',disparity)


# ## Ex.2 Iterating Through States

# In[ ]:


disparities=[]
for state in statedict.keys():
    print("")
    pb1=float(readCSV[statedict[state]+1][6])/float(readCSV[statedict[state]+1][2].replace(',', ''))
    print('the probability that someone is killed by an officer in',state,'given theyre black is',pb1)
    rate1=pb1*100000
    print('the rate of killing per 100k is thus',rate1)
    
    whitepop=float(readCSV[statedict[state]+1][17])*int(readCSV[statedict[state]+1][1].replace(',', ''))
    pb2=float(readCSV[statedict[state]+1][11])/(whitepop)
    print('the probability that someone is killed by an officer',state,'given theyre white is',pb2)
    rate2=pb2*100000
    print('the rate of killing per 100k is thus',rate2)
    
    pb3=(float(readCSV[statedict[state]+1][6])+float(readCSV[statedict[state]+1][7])
    +float(readCSV[statedict[state]+1][8])+float(readCSV[statedict[state]+1][9])
    +float(readCSV[statedict[state]+1][10]))/(int(readCSV[statedict[state]+1][1].replace(',', ''))-whitepop)
    print('the probability that someone is killed by an officer given theyre not white is',pb3)
    print('the rate of killing per 100k is thus',pb3*100000)
    rate3=pb3*100000
    
    bwdisparity =rate1-rate2
    print('the black-white disparity is ',bwdisparity)
    disparity=rate3-rate2
    disparities.append(disparity)
    print('the disparity is ',disparity)
    print("")


# In[ ]:


avg=np.sum(disparities)/len(disparities)
print('national disparity is',avg)


# # Part III: Aggregate Data and Joint Sampling

# In[ ]:


with open('aggregate.csv') as csvfile:
    csvthing = csv.reader(csvfile, delimiter=',')
    total=0
    count=0
    readCSV=[]
    for row in csvthing:
        readCSV.append(row)
        
#dictify row 1
    querydict={}
    labels=readCSV[0]
    for el in labels:
        name=el
        querydict[name]=count
        count=count+1
    count=0

    print(querydict)
        


# In[ ]:


n=1000000
wcount=0
for row in readCSV:
        if row[3]=='White' or row[3]=='white':
            wcount=wcount+1
            
bcount=0
for row in readCSV:
        if row[3]=='Black' or row[3]=='black':
            bcount=bcount+1

ncount=0
for row in readCSV:
        if row[3]=='Unknown Race' or row[3]=='?':
            ncount=ncount+1
tcount=len(readCSV)-1

#prob of police homicide
probPoliceHomicide= tcount/327000000 #P(H)
ratePoliceHomicide=n*probPoliceHomicide 

#prob of police homicide involving someone white
probWHomicide= wcount/327000000 #P(W,H)
rateWHomicide=n*probWHomicide
probHomicidegW=probWHomicide/(0.607) #P(H|W)
rateHomicidegW=n*probHomicidegW

#prob of shooting someone black
probBHomicide= bcount/327000000 #P(B,H)
rateBHomicide=n*probBHomicide
probHomicidegB=probBHomicide/(0.14) #P(B|H)
rateHomicidegB=n*probHomicidegB

print(rateHomicidegW,'police homicides of white people per million in America')
print(rateHomicidegB,'police homicides of black people per million in America')


#prob of police homicide involving someone black and unarmed
ucount=0
for row in readCSV:
    if row[18]=='Unarmed' or row[18]=='unarmed':
        ucount=ucount+1
probUH=ucount/327000000

probHgU=probUH/probPoliceHomicide

ucount=0
for row in readCSV:
    if (row[18]=='Unarmed' or row[18]=='unarmed') and (row[3]=='Black' or row[3]=='black'):
        ucount=ucount+1
probBUH=ucount/327000000

probBgUH=probBUH/probHgU

ucount=0
for row in readCSV:
    if row[18]=='Armed' or row[18]=='Allegedly Armed':
        ucount=ucount+1
probAH=ucount/327000000

ucount=0
for row in readCSV:
    if (row[18]=='Armed' or row[18]=='Allegedly Armed') and (row[3]=='Black' or row[3]=='black'):
        ucount=ucount+1
probBAH=ucount/327000000

probHgA=probAH/probPoliceHomicide
probBgAH=probBAH/probHgA

probB=.14
probU=.99 # approximated, 3 million Americans carry guns daily, more probably carry occasionally/illegally
#this data obviously doesn't exist, so i had to make a generous guess

probBgU=probB*probU

probHgBU=(probBgUH*probHgU)/probBgU
print('probability of a police homicide, given a black unarmed victim is',probHgBU)
print(probHgBU*1000000,'homicides per million')


#prob of police homicide involving someone white and unarmed
ucount=0
for row in readCSV:
    if (row[18]=='Unarmed' or row[18]=='unarmed') and (row[3]=='White' or row[3]=='white'):
        ucount=ucount+1
probWUH=ucount/327000000

probWgUH=probWUH/probHgU

ucount=0
for row in readCSV:
    if (row[18]=='Armed' or row[18]=='Allegedly Armed') and (row[3]=='White' or row[3]=='white'):
        ucount=ucount+1
probWAH=ucount/327000000

probWgAH=probWAH/probHgA

probW=.613

probWgU=probW*probU

probHgWU=(probWgUH*probHgU)/probWgU
print('probability of a police homicide, given a white unarmed victim is',probHgWU)
print(probHgWU*1000000,'homicides per million')

print('the disparity is now even larger as the rate of murder is around 4 times larger for unarmed black people')

probHgB=probHomicidegB #rewriting variable names for clarity later on
probHgW=probHomicidegW
probHgBA=probHgB-probHgBU # since probHgB=probHgBU+probHgBA (armed and unarmed are mutually exclusive+exhaustive)
probHgWA=probHgW-probHgWU # above

def probHomicide(b,w,u):
    if u==1:
        if b==1:
            return probHgBU
        elif w==1:
            return probHgWU
        else:
            return probHgU
    else:
        if b==1:
            return probHgBA
        elif w==1:
            return probHgWA
        else:
            return probHgA


# We can visualize these distributions to show how large the disparity is (black is red, white is blue):

# In[ ]:


from scipy.stats import poisson

w=poisson(3.27)
b=poisson(15)
arr=[]
for num in range(-5,35):
    arr.append(w.pmf(num))
prob1 = w.pmf(5)
plt.grid(True)
plt.ylabel('Probability ')
plt.xlabel('# of Annual Homicides of Unarmed People')
plt.title('Probability Distribution Curve')
plt.plot(arr, linewidth=2.0)

arr2=[]
for num in range(-5,35):
    arr2.append(b.pmf(num))
prob2 = b.pmf(12)
plt.grid(True)
plt.plot(arr2, linewidth=2.0)


# In[ ]:


#calculating multivariable probabilities
counter=0
for row in readCSV:
    if ("Charged" in str(row[14])) or ("indicted" in str(row[14])) or ("Murder" in str(row[14])) or ("Criminal" in str(row[14])) or ("Convicted" in str(row[14])):
        counter=counter+1
pCH=counter/327000000
probCgH=pCH/probPoliceHomicide
print(probCgH, 'is the probability that an officer is charged with homicide given a police homicide')

#assumes homicide occurs, since makeSample only runs this if homicide=1
for row in readCSV:
    c=("Charged" in str(row[14])) or ("indicted" in str(row[14])) or ("Murder" in str(row[14])) or ("Criminal" in str(row[14])) or ("Convicted" in str(row[14]))
    if c and (row[3]=='Black' or row[3]=='black') and (row[18]=='Unarmed' or row[18]=='unarmed'):
        counter=counter+1
probCBUH=counter/327000000

for row in readCSV:
    c=("Charged" in str(row[14])) or ("indicted" in str(row[14])) or ("Murder" in str(row[14])) or ("Criminal" in str(row[14])) or ("Convicted" in str(row[14]))
    if c and (row[3]=='White' or row[3]=='white') and (row[18]=='Unarmed' or row[18]=='unarmed'):
        counter=counter+1
probCWUH=counter/327000000

for row in readCSV:
    c=("Charged" in str(row[14])) or ("indicted" in str(row[14])) or ("Murder" in str(row[14])) or ("Criminal" in str(row[14])) or ("Convicted" in str(row[14]))
    if c and (row[18]=='Unarmed' or row[18]=='unarmed'):
        counter=counter+1
probCUH=counter/327000000

for row in readCSV:
    c=("Charged" in str(row[14])) or ("indicted" in str(row[14])) or ("Murder" in str(row[14])) or ("Criminal" in str(row[14])) or ("Convicted" in str(row[14]))
    if c and (row[3]=='Black' or row[3]=='black') and (row[18]=='Allegedly Armed' or row[18]=='Armed'):
        counter=counter+1
probCBAH=counter/327000000

for row in readCSV:
    c=("Charged" in str(row[14])) or ("indicted" in str(row[14])) or ("Murder" in str(row[14])) or ("Criminal" in str(row[14])) or ("Convicted" in str(row[14]))
    if c and (row[3]=='White' or row[3]=='white') and (row[18]=='Allegedly Armed' or row[18]=='Armed'):
        counter=counter+1
probCWAH=counter/327000000

for row in readCSV:
    c=("Charged" in str(row[14])) or ("indicted" in str(row[14])) or ("Murder" in str(row[14])) or ("Criminal" in str(row[14])) or ("Convicted" in str(row[14]))
    if c and (row[18]=='Allegedly Armed' or row[18]=='Armed'):
        counter=counter+1
probCAH=counter/327000000

#using bayes rule, g means given
probCgBUH=probCBUH/(probBUH)
probCgWUH=probCWUH/(probWUH)
probCgUH=probCUH/probUH

#using bayes rule, g means given
probCgBAH=probCBAH/(probBAH)
probCgWAH=probCWAH/(probWAH)
probCgAH=probCAH/probAH

#again assumes homicide occurs, since makeSample only runs this if homicide=1
def probCharges(black,white,unarmed):
    counter=0
    if unarmed==1:
        #P(C|B=1,W=0,U=1,H)=P(C,B=1,W=0,U=1)/P(B=1,W=0,U=1,H)
        if black==1:
            return probCgBUH
        #P(C,B=0,W=1,U=1,H)/P(B=0,W=1,U=1,H)
        elif white==1:
            return probCgWUH
        #P(C,B=0,W=0,U=1,H)/P(B=0,W=0,U=1,H)
        else:
            return probCgUH
    else:
        #P(C,B=1,W=0,U=0,H)/P(B=1,W=0,U=0,H)
        if black==1:
            return probCgBAH
        #P(C,B=0,W=1,U=0,H)/P(B=0,W=1,U=0,H)
        elif white==1:
            return probCgWAH
        #P(C,B=0,W=0,U=0,H)/P(B=0,W=0,U=0,H)
        else:
            return probCgAH


# In[ ]:


print(probCgBUH, 'is the probability of an officer being charged given they shot someone black and unarmed')
print(probCgWUH, 'is the probability of an officer being charged given they shot someone white and unarmed')


# In[ ]:


#prob conviction
counter=0
for row in readCSV:
    if ("Murder" in str(row[14])) or ("Criminal" in str(row[14])) or ("Convicted" in str(row[14])):
        counter=counter+1
pVH=counter/327000000
probVgH=pVH/probPoliceHomicide
print(probVgH, 'is the probability that an officer is convicted of homicide given a homicide')

#assumes homicide occurs, since makeSample only runs this if homicide=1
for row in readCSV:
    c=("Murder" in str(row[14])) or ("Criminal" in str(row[14])) or ("Convicted" in str(row[14]))
    if c and (row[3]=='Black' or row[3]=='black') and (row[18]=='Unarmed' or row[18]=='unarmed'):
        counter=counter+1
probVCBUH=counter/327000000

for row in readCSV:
    c=("Murder" in str(row[14])) or ("Criminal" in str(row[14])) or ("Convicted" in str(row[14]))
    if c and (row[3]=='White' or row[3]=='white') and (row[18]=='Unarmed' or row[18]=='unarmed'):
        counter=counter+1
probVCWUH=counter/327000000

for row in readCSV:
    c=("Murder" in str(row[14])) or ("Criminal" in str(row[14])) or ("Convicted" in str(row[14]))
    if c and (row[18]=='Unarmed' or row[18]=='unarmed'):
        counter=counter+1
probVCUH=counter/327000000

for row in readCSV:
    c=("Murder" in str(row[14])) or ("Criminal" in str(row[14])) or ("Convicted" in str(row[14]))
    if c and (row[3]=='Black' or row[3]=='black') and (row[18]=='Allegedly Armed' or row[18]=='Armed'):
        counter=counter+1
probVCBAH=counter/327000000

for row in readCSV:
    c=("Murder" in str(row[14])) or ("Criminal" in str(row[14])) or ("Convicted" in str(row[14]))
    if c and (row[3]=='White' or row[3]=='white') and (row[18]=='Allegedly Armed' or row[18]=='Armed'):
        counter=counter+1
probVCWAH=counter/327000000

for row in readCSV:
    c=("Murder" in str(row[14])) or ("Criminal" in str(row[14])) or ("Convicted" in str(row[14]))
    if c and (row[18]=='Allegedly Armed' or row[18]=='Armed'):
        counter=counter+1
probVCAH=counter/327000000

#using bayes rule, g means given
probVgCBUH=probVCBUH/(probCBUH)
probVgCWUH=probVCWUH/(probCWUH)
probVgCUH=probVCUH/probUH

#using bayes rule, g means given
probVgCBAH=probVCBAH/(probCBAH)
probVgCWAH=probVCWAH/(probCWAH)
probVgCAH=probVCAH/probCAH

#again assumes homicide occurs, since makeSample only runs this if homicide=1
def probConviction(black,white,unarmed):
    counter=0
    if unarmed==1:
        #P(C|B=1,W=0,U=1,H)=P(C,B=1,W=0,U=1,H)/P(B=1,W=0,U=1,H)
        if black==1:
            return probVgCBUH
        #P(C|B=0,W=1,U=1,H)=P(C,B=0,W=1,U=1,H)/P(B=0,W=1,U=1,H)
        elif white==1:
            return probVgCWUH
        #P(C|U=1,H)=P(C,U=1,H)/P(U=1,H)
        else:
            return probVgCUH
    else:
        #P(C,B=1,W=0,U=0,H)/P(B=1,W=0,U=0,H)
        if black==1:
            return probVgCBAH
        #P(C,B=0,W=1,U=0,H)/P(B=0,W=1,U=0,H)
        elif white==1:
            return probVgCWAH
        #P(C,U=0,H)/P(U=0,H)
        else:
            return probVgCAH


# In[ ]:


print(probVgCBUH, 'is the probability of an officer being convicted given they are charged for shooting someone black and unarmed')
print(probVgCWUH, 'is the probability of an officer being convicted given they are charged for shooting someone white and unarmed')


# It becomes very difficult to model with different parameters after the addition of the charges event, so we need some way to model this Bayesian network of events. Joint sampling works well.

# In[ ]:


import random

# joint sampling 
def bern(p):
    event = random.random() < p
    if event: return 1
    else: return 0
    
def makeSample():
    observation=[]
    unarmed = bern(0.99)
    white=bern(.613)
    if white!=1:
        black = bern(0.14/(1-0.613)) #P(B|not-white)=P(B,non-white)/P(non-white)=P(B)/P(non-white)
    else:
        black=0
    homicide=bern(probHomicide(black,white,unarmed))
    if homicide==1:
        charges=bern(probCharges(black,white,unarmed))
    else:
        charges=0
    if charges==1:
        conviction=probConviction(black,white,unarmed)
    else:
        conviction=0
    observation = [unarmed,black,white,homicide,charges,conviction]
    return observation
    
def makeSamples(n):
    samples=[]
    for i in range(n):
        samples.append(makeSample())
    return samples        

samples=makeSamples(32700000) #1/10 of the US population


# In[ ]:


def probSample(a,an): #P(a=an)
    counter=0
    for s in samples:
        if (s[a]==an):
            counter=counter+1
    count = counter
    total=len(samples)
    return count/total

def probSample1(a,b,an,bn): #P(a=an|b=bn)
    keptsamples = []
    for s in samples:
        if (s[b]==bn):
            keptsamples.append(s)
    total = len(keptsamples)
    count=0
    for s in keptsamples:
        if s[a]==an:
            count=count+1
    return count/total

def probSample2(a,b,c,an,bn,cn): #P(a=an|b=bn,c=cn)
    keptsamples = []
    for s in samples:
        if (s[b]==bn and s[c]==cn):
            keptsamples.append(s)
    total = len(keptsamples)
    count=0
    for s in keptsamples:
        if s[a]==an:
            count=count+1
    return count/total

def probSample3(a,b,c,d,an,bn,cn,dn): #P(a|b,c,d)
    keptsamples = []
    for s in samples:
        if (s[b]==bn and s[c]==cn and s[d]==dn):
            keptsamples.append(s)
    total = len(keptsamples)
    count=0
    for s in keptsamples:
        if s[a]==an:
            count=count+1
    return count/total


# Initially running this below cell gave only a 5% white population because i didn't realize that there should be arrows between black and white in my Bayesian network (since these random variables obviously affect each other). After fixing that, I adjusted my makeSamples() method using Bayes theorem as shown above and then the proportions started to make sense with actual US census data.

# In[ ]:


print('portion of sample that is black',probSample(1,1))
print('portion of sample that is white',probSample(2,1))


# The joint sampled model mimics the close, but regardless disparate, rates of officers getting charged when they shoot unarmed people. The fact that there are 0 convictions is unfamiliar though. However, probably due to the smaller sample size than the actual population of the US and the ridiculously small amount of officers that get convicted, with a sample with a size of 1/10th the US' population, it makes sense that no one gets convicted. Doing the math out, I realized that this made sense so there was no issue with my algorithm.

# In[ ]:


print('probability of charges given police homicide of an unarmed black person is ',probSample3(4,1,3,0,1,1,1,1))
print('probability of charges given police homicide of an unarmed white person is ',probSample3(4,2,3,0,1,1,1,1))
print('probability of conviction',probSample(5,1))


# The joint sampled model also shows a 4x disparity between white and black people so it succeeds in approximating disparities in the US.

# In[ ]:


p1=probSample2(3,1,0,1,1,1)
p2=probSample2(3,2,0,1,1,1)
print('probability of a police homicide of an unarmed black person is ',p1)
print('thus the rate per million is', p1*1000000)
print('probability of a police homicide of an unarmed white person is ',p2)
print('thus the rate per million is', p2*1000000)


# # FUTURE WORK

# # P-values?

# In[ ]:


# we can now calcualte p-values for a national disparity?
# that would require two generated samples of b/w populations
# and drawing out 
import random
def pvalue(universalSample,hopes):
    s=universalSample
    n=wcount
    m=tcount-wcount
    count=0
    for i in range(10000):
        r1=random.choices(s,k=n)
        sum1=0
        for i in r1:
            sum1=sum1+int(i)
        r2=random.choices(s,k=m)
        sum2=0
        for i in r2:
            sum2=sum2+int(i)
        rate1=sum1/float(n)
        rate2=sum2/float(m)
        ratediff=abs(rate1-rate2)
        if ratediff>hopes:
            count=count+1
    return count/10000

print(pvalue(np.ones(tcount),13.214192025732315))


# # Creating a GUI for a Probability Calculator

# In[ ]:


import PySimpleGUI as sg

layout = [ [sg.Txt('Enter values to calculate')],      
           [sg.In(size=(8,1), key='numerator')],      
           [sg.Txt('_'  * 10)],      
           [sg.In(size=(8,1), key='denominator')],      
           [sg.Txt('', size=(8,1), key='output')  ],      
           [sg.Button('Calculate', bind_return_key=True)]]

window = sg.Window('Math', layout)

while True:      
    event, values = window.Read()

    if event is not None:      
        try:      
            numerator = float(values['numerator'])      
            denominator = float(values['denominator'])      
            calc = numerator / denominator      
        except:      
            calc = 'Invalid'

        window.Element('output').Update(calc)      
    else:      
        break

