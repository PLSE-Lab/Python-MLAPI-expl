# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy # linear algebra
import pandas # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
from datetime import datetime

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

first_file="../input/kiva_loans.csv"
Loans=pandas.read_csv(first_file)
print(Loans.head())

second_file ="../input/kiva_mpi_region_locations.csv"
MPIregion = pandas.read_csv(second_file)

third_file = "../input/loan_theme_ids.csv"
LoanThemes = pandas.read_csv(third_file)

fourth_file = "../input/loan_themes_by_region.csv"
ThemesxRegion = pandas.read_csv(fourth_file)




# Any results you write to the current directory are saved as output.

###finding the mutual information gained between selected features and MPI
############

#firstJoin = pandas.merge(Loans, MPIregion, left_on='region', right_on='region')
#secondJoin = pandas.merge(firstJoin, themesXregion, left_on='region', right_on='region')

#third join not used, but should be used for better results
#thirdJoin = pandas.merge(secondJoin, LoanThemes, left_on='id', right_on='id' )

startTime1 = datetime.now()




#looking at MPI distribution
sortMPI = numpy.sort(MPIregion.loc[:, 'MPI'])
IncreasingList = list(range(sortMPI.shape[0]))
#print('Plotting sorted MPI :', matplotlib.pyplot.scatter(IncreasingList, sortMPI)) bad idea - gave no useful info lol

Activities = set(Loans.loc[:, 'activity'])
Sectors = set(Loans.loc[:, 'sector'])
Repayment_intervals = set(Loans.loc[:, 'repayment_interval'])
Loan_amounts = set(Loans.loc[:, 'loan_amount'])
Bor_Genders = set(Loans.loc[:, 'borrower_genders'])


NumAct = {}

for i in range(len(Loans.loc[:, 'activity'])):
    string = Loans.loc[i, 'activity']
    if string in NumAct.keys():
        NumAct[string] = NumAct[string] + 1
    else:
        NumAct[string] = 1
    
total = sum(NumAct.values(), 0.0)
NumActPct = {k: v / total for k, v in NumAct.items()}

#################################################################

NumReg = {}

for i in range(len(Loans.loc[:, 'region'])):
    string = Loans.loc[i, 'region']
    if string in NumReg.keys():
        NumReg[string] = NumReg[string] + 1
    else:
        NumReg[string] = 1
    
total = sum(NumReg.values(), 0.0)
NumRegPct = {k: v / total for k, v in NumReg.items()}


############################################################################

NumSector = {}

for i in range(len(Loans.loc[:, 'sector'])):
    string = Loans.loc[i, 'sector']
    if string in NumSector.keys():
        NumSector[string] = NumSector[string] + 1
    else:
        NumSector[string] = 1
 
total = sum(NumSector.values(), 0.0)
NumSectorPct = {k: v / total for k, v in NumSector.items()}

###################################################################

NumInterval = {}

for i in range(len(Loans.loc[:, 'repayment_interval'])):
    string = Loans.loc[i, 'repayment_interval']
    if string in NumInterval.keys():
        NumInterval[string] = NumInterval[string] + 1
    else:
        NumInterval[string] = 1
        
total = sum(NumInterval.values(), 0.0)
NumIntervalPct = {k: v / total for k, v in NumInterval.items()}
 ##########################################################################

LoanAmount = {}

for i in range(len(Loans.loc[:, 'loan_amount'])):
    string = Loans.loc[i, 'loan_amount']
    if string in LoanAmount.keys():
        LoanAmount[string] = LoanAmount[string] + 1
    else:
        LoanAmount[string] = 1
        
total = sum(LoanAmount.values(), 0.0)
LoanAmountPct = {k: v / total for k, v in LoanAmount.items()}

########################################################################## 

BorGender = {}

for i in range(len(Loans.loc[:, 'borrower_genders'])):
    string = Loans.loc[i, 'borrower_genders']
    if string in BorGender.keys():
        BorGender[string] = BorGender[string] + 1
    else:
        BorGender[string] = 1
        
total = sum(BorGender.values(), 0.0)
BorGenderPct = {k: v / total for k, v in BorGender.items()}

###########################################################################      

Tags = set(Loans.loc[:, 'tags'])
        

time1 = datetime.now() - startTime1
print('The time taken to run part1 is - ', datetime.now() - startTime1)
###############################################################################################################

startTime2 = datetime.now()

MPIrange = max(MPIregion.loc[:, 'MPI']) - min(MPIregion.loc[:, 'MPI'])
step = MPIrange/6
MPIsort = numpy.sort(MPIregion.loc[:, 'MPI'], axis= -1, kind='mergesort')
sortedMPI = MPIsort[numpy.isfinite(MPIsort)]

MPIpdf = [0]
MPIused = [sortedMPI[0]]

funcstart = 0

for x in range(len(sortedMPI) - 1):
    newMPI = sortedMPI[x]
    newAdded = 0
    if x <= 0:
        MPIused[0] = newMPI
        MPIpdf[0] = 0
    else:
        MPIused = numpy.vstack([MPIused, newMPI])
        MPIpdf = numpy.vstack([MPIpdf, 0])
    for y in range(len(MPIused) - funcstart):
        func = 1/(1+(numpy.power((numpy.power((MPIpdf[y] - newMPI),2)* 4/step),3))) 
        if func <= 0.1:
            funcstart = y
            
        MPIpdf[y] = MPIpdf[y] + func
        newAdded = newAdded + func
        
    MPIpdf[x] = MPIpdf[x] + newAdded
    
MPIpdf = MPIpdf / sum(MPIpdf)[0]

MPIpdf_dict = {}
for x in range(len(MPIused)):
    MPIpdf_dict[MPIused[x][0]] = MPIpdf[x][0]

time2 = datetime.now() - startTime2    
print('The time taken to run part2 is - ', datetime.now() - startTime2 )

#print('Plotting MPIpdf :',matplotlib.pyplot.scatter(MPIused, MPIpdf*15))
    
#####################################################################################################    
        
##################
###################################################################
##################
#######################################################################################################
#compute Mutual information now

#mutual information for activity and MPI ---- 
#iterate through set of activities
startTime3 = datetime.now()
Act_MPI_MI = 0
for n in Activities:
    newSet = set(Loans.loc[Loans['activity'] == n, 'region'])
    for m in newSet:
        PctBoth = len(Loans.loc[(Loans['activity'] == n) & (Loans['region'] == m)]) / len(Loans.loc[:])
        PctN = NumActPct[n]
        PctM = NumRegPct[m]
        
        if PctBoth > 0:
            PctBoth2 = PctBoth
        elif PctBoth <= 0:
            PctBoth2 = 0.00000000000000000000000000000000000001
        
        MI = PctBoth * numpy.log(PctBoth2/(PctN*PctM)) #why is this dividing by zero?
        Act_MPI_MI = Act_MPI_MI + MI

############
##Act_MPI_MI = 1.3375874918728894 ------------------ correct
############
time3 = datetime.now() - startTime3
print('The time it took to find the Act_MPI_MI is: ', time3)
print(Act_MPI_MI + ' - this is the Act_MPI_MI')

##############################################################################################
# compute mutual information for Sector and MPI ----------takes 1:25:00.00 or so

startTime4 = datetime.now()
Sec_MPI_MI = 0
for n in Sectors:
    newSet = set(Loans.loc[Loans['sector'] == n, 'region'])
    for m in newSet:
        PctBoth = len(Loans.loc[(Loans['sector'] == n) & (Loans['region'] == m)]) / len(Loans.loc[:])
        PctN = NumSectorPct[n]
        PctM = NumRegPct[m]
        
        if PctBoth > 0:
            PctBoth2 = PctBoth
        elif PctBoth <= 0:
            PctBoth2 = 0.00000000000000000000000000000000000001
        
        MI = PctBoth * numpy.log(PctBoth2/(PctN*PctM)) #why is this dividing by zero?
        Sec_MPI_MI = Sec_MPI_MI + MI

##########################
#Sec_MPI_MI = 0.64642748300153174  -------------- correct? 
##################    
time4 = datetime.now() - startTime4
print('The time it took to find the Sec_MPI_MI is: ', time4)
print(Sec_MPI_MI, ' - this is the Sec_MPI_MI')

################################################################################################
# compute mutual information for Rep_Int and MPI ---------takes 00:38:00.00 or so

startTime5 = datetime.now()
Rep_Int_MPI_MI = 0
for n in Repayment_intervals:
    newSet = set(Loans.loc[Loans['repayment_interval'] == n, 'region'])
    for m in newSet:
        PctBoth = len(Loans.loc[(Loans['repayment_interval'] == n) & (Loans['region'] == m)]) / len(Loans.loc[:])
        PctN = NumIntervalPct[n]
        PctM = NumRegPct[m]
        
        if PctBoth > 0:
            PctBoth2 = PctBoth
        elif PctBoth <= 0:
            PctBoth2 = 0.00000000000000000000000000000000000001
        
        MI = PctBoth * numpy.log(PctBoth2/(PctN*PctM)) #why is this dividing by zero?
        Rep_Int_MPI_MI = Rep_Int_MPI_MI + MI

######################
#Rep_Int_MPI_MI = 0.67659162860931432   -------------------------------- correct?
######################        
time5 = datetime.now() - startTime5
print('The time it took to find the Rep_Int_MPI_MI is: ', time5)
print(Rep_Int_MPI_MI, ' - this is the Rep_Int_MPI_MI')
############################################################################################

#compute MI of loan_amount and MPI

startTime6 = datetime.now()
LoanAmount_MPI_MI = 0
for n in Loan_amounts:
    newSet = set(Loans.loc[Loans['loan_amount'] == n, 'region'])
    for m in newSet:
        PctBoth = len(Loans.loc[(Loans['loan_amount'] == n) & (Loans['region'] == m)]) / len(Loans.loc[:])
        PctN = LoanAmountPct[n]
        PctM = NumRegPct[m]
        
        if PctBoth > 0:
            PctBoth2 = PctBoth
        elif PctBoth <= 0:
            PctBoth2 = 0.00000000000000000000000000000000000001
        
        MI = PctBoth * numpy.log(PctBoth2/(PctN*PctM)) #why is this dividing by zero?
        LoanAmount_MPI_MI = LoanAmount_MPI_MI + MI

####################
#LoanAmount_MPI_MI = 1.1778368365269787 -------------------------------- correct?
#######################
time6 = datetime.now() - startTime6
print('The time it took to find the LoanAmount_MPI_MI is: ', time6)
print(LoanAmount_MPI_MI, ' - this is the LoanAmount_MPI_MI')

#compute MI of borrower_gender and MPI

startTime7 = datetime.now()
BorGender_MPI_MI = 0
for n in Bor_Genders:
    newSet = set(Loans.loc[Loans['borrower_genders'] == n, 'region'])
    for m in newSet:
        PctBoth = len(Loans.loc[(Loans['borrower_genders'] == n) & (Loans['region'] == m)]) / len(Loans.loc[:])
        PctN = BorGenderPct[n]
        PctM = NumRegPct[m]
        
        if PctBoth > 0:
            PctBoth2 = PctBoth
        elif PctBoth <= 0:
            PctBoth2 = 0.00000000000000000000000000000000000001
        
        MI = PctBoth * numpy.log(PctBoth2/(PctN*PctM)) #why is this dividing by zero?
        BorGender_MPI_MI = BorGender_MPI_MI + MI

################
#BorGender_MPI_MI = 0.74761859535471997 -------------------------------- correct?
#################
time7 = datetime.now() - startTime7
print('The time it took to find the BorGender_MPI_MI is: ', time7)
print(BorGender_MPI_MI, ' - this is the Borrrowers Gender + MPI mutial info')
