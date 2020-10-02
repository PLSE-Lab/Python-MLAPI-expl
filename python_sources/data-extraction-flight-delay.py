"this program serves to extract flight delay data from .csv files"
"the data will be place in one list, with each flight as separate item"
"each flight consists of a number of message lines"
"each message has 19 inputs"
import numpy as np
from math import *
import os
import csv
from multiprocessing import Pool
import time
import pandas as pd

starttime=time.time()

#name of data file

fname= '../input/InputData.csv'

#list with separate flights as inputs
flightlist=[]
#read data from .csv file
f=open(fname,'rt')
reader=csv.reader(f)

#read each line, assign flight numbers to tmplist
tmplist=[]

for row in reader:
    tmplist.append(row)
tmplist2 = []
f.close()

for i in range(2, len(tmplist)):
    
    
    if  tmplist[i-1][10]=='AMS' and (tmplist[i-1][4]=='.' or tmplist[i-1][5]=='.') or tmplist[i-1][9]=='AMS' and ( tmplist[i-1][2]=='.' or tmplist[i-1][3]=='.') or tmplist[i-1][12]=='.'  or tmplist[i-1][16]=='.':
        continue      
    tmplist2.append(tmplist[i-1])
    if i== (len(tmplist)-1):
        tmplist2.append(tmplist[i])
        flightlist.append(tmplist2)
    else:
        
        flightlist.append(tmplist2)
        
        tmplist2 = []

def Date(fllist):
    Dates = []
    datefl = []
    datemes = []
    a = [1,2,3,4,5,12,16]
    mon = ['NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT']
    monlen = [0,30,61,92,120,151,181,212,242,273,304,334]
    for i in range(len(fllist)):
                 
        month = fllist[i][0][1][2:5]
        day = int(fllist[i][0][1][0:2])-22
        mnnr = mon.index(month)
        day = day + monlen[mnnr]
        Dates.append(day)
    return Dates

def SchDep(flightlist):
    schdep = []
    for i in range(len(flightlist)):
        time = (flightlist[i][0][2][8:13])
        if flightlist[i][0][2]=='.':
            schdep.append('N.A.')
        else:
            minutes = int(time[0:2])*60 + int(time[3:5])
            schdep.append(minutes)
    return schdep

def ActDep(flightlist):
    actdep = []
    for i in range(len(flightlist)):
        time = flightlist[i][0][3][8:13]
        if flightlist[i][0][3]=='.':
            actdep.append('N.A.')
        else:
            minutes = int(time[0:2])*60 + int(time[3:5])
            actdep.append(minutes)
    return actdep

def SchArr(flightlist):
    scharr = []
    for i in range(len(flightlist)):
        time = (flightlist[i][0][4][8:13])
        if flightlist[i][0][4]=='.':
            scharr.append('N.A.')
        else:
            minutes = int(time[0:2])*60 + int(time[3:5])
            scharr.append(minutes)
    return scharr

def ActArr(flightlist):
    actarr = []
    for i in range(len(flightlist)):
        
        time = (flightlist[i][0][5][8:13])
        if flightlist[i][0][5]=='.':
            actarr.append('N.A.')
        else:
            minutes = int(time[0:2])*60 + int(time[3:5])
            actarr.append(minutes)
    return actarr

def Airline(flightlist):
    airline = []
    for i in range(len(flightlist)):
        airplane = flightlist[i][0][0][0:2]
        airline.append(airplane)
    return airline

def DepAir(flightlist):
    depair = []
    for i in range(len(flightlist)):
        depstn = flightlist[i][0][9]
        depair.append(depstn)
    return depair

def ArrAir(flightlist):
    arrair = []
    for i in range(len(flightlist)):
        arrstn = flightlist[i][0][10]
        arrair.append(arrstn)
    return arrair

def AirType(flightlist):
    airtype = []
    for i in range(len(flightlist)):
        airtyp = flightlist[i][0][7]
        airtype.append(airtyp)
    return airtype

def PredTime(flightlist):
    tottime = []
    for i in range(len(flightlist)):
        fltime = []
        for j in range(len(flightlist[i])):
            predtime = flightlist[i][j][16][8:13]
            
            minutes = int(predtime[0:2])*60 + int(predtime[3:5])
            fltime.append(minutes)
        tottime.append(fltime)
    return tottime

def MessTime(flightlist):
    tottime = []
    for i in range(len(flightlist)):
        fltime = []
        for j in range(len(flightlist[i])):
            messtime = flightlist[i][j][12][8:13]
            minutes = int(messtime[0:2])*60 + int(messtime[3:5])
            fltime.append(minutes)
        tottime.append(fltime)
    return tottime

dates = Date(flightlist)
schdep =  SchDep(flightlist)
actdep =  ActDep(flightlist)
scharr = SchArr(flightlist)
actarr = ActArr(flightlist)
airline = Airline(flightlist)
depair = DepAir(flightlist)
arrair = ArrAir(flightlist)
airtype = AirType(flightlist)
predtime = PredTime(flightlist)
messtime = MessTime(flightlist)

def Error(flightlist):
    toterror = []
    for i in range(len(flightlist)):
        
        flerror = []
        for j in range(len(flightlist[i])):
            
            if depair[i]=='AMS':
                errtime =  int(actdep[i]) - int(predtime[i][j])
            if arrair[i]=='AMS':
                errtime =  int(actarr[i]) - int(predtime[i][j])
            flerror.append(errtime)
        toterror.append(flerror)
    return toterror
            
error = Error(flightlist)

endlist = []
listmes = []
listflight = []
for i in range(len(flightlist)):
    for j in range(len(flightlist[i])):
        listmes.append(dates[i])
        listmes.append(schdep[i])
        listmes.append(actdep[i])
        listmes.append(scharr[i])
        listmes.append(actarr[i])
        listmes.append(airline[i])
        listmes.append(depair[i])
        listmes.append(arrair[i])
        listmes.append(airtype[i])
        listmes.append(predtime[i][j])
        listmes.append(messtime[i][j])
        listmes.append(error[i][j])
        endlist.append(listmes)
        listmes = []





        
