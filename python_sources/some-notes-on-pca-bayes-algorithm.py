#!/usr/bin/env python
# coding: utf-8

# The problem of Native Bayesian algorithm is : all features must be relatively independent. Originally, I directly use bayesian algorithm to classify leaves. the code and result as follows:

# In[ ]:


# apply bayesian model for calculating the probability of sample
def BayesClassify0(testData,dicSubsetMean,dicSubsetVar,dicSubsetPbi):
	dicTestDataBayes = {}
	n = np.shape(testData)[1]
	#
	for key in dicSubsetMean.keys():
		dataMean = dicSubsetMean[key][0]
		dataVar = dicSubsetVar[key][0]
		PBi = dicSubsetPbi[key]
		testData = testData[0]
		PABi = [0]
		for j in range(n):
			if dataVar[0,j]==0:
				break
			PAjBi = np.exp(-pow((testData[0,j]-dataMean[0,j]),2)/(2*dataVar[0,j]))/np.sqrt(2*3.1415*dataVar[0,j])
			if PAjBi>0:
				PABi.append(math.log(PAjBi,2))
		PBiA = sum(PABi) + PBi
		dicTestDataBayes[key] = PBiA
	#
	sortPBiA = sorted(dicTestDataBayes.iteritems(),key=operator.itemgetter(1),reverse=True)
	return sortPBiA[0][0]


# Total test samples:49 	avgErrCount:38 	avgErrRate:0.789116
# 
# the above result is bad, I think the reason is that some original features are locally relevant.
# so I allpy PCA to make features irrelevant, and the run bayesian algorithm again.
# the code and results as follows:

# In[ ]:


# PCA
def pca(dataSet,topNfeat=9999):
	meanVals = np.mean(dataSet,axis=0)
	dataRemoved = dataSet-meanVals
	covMat = np.cov(dataRemoved,rowvar=0)
	#
	eigVals,eigVects = np.linalg.eig(np.mat(covMat))
	eigValInd = np.argsort(eigVals)
	eigValInd = eigValInd[:-(topNfeat+1):-1]
	redEigVecs = eigVects[:,eigValInd]
	#
	reduceSet = dataRemoved*redEigVecs
	return reduceSet,eigVals,redEigVecs

def contribution(eigVals,tops):
	total = sum(eigVals)
	for i in range(tops):
		conYi = eigVals[i]/total
		print "No.",i+1," contribution rate:",conYi
	sconYi = sum(eigVals[:i+1])/total
	print "the top ",i+1," contribution rate:",sconYi,"\n"


# No. 1  contribution rate: 0.149800041582
# 
# No. 2  contribution rate: 0.101889003956
# 
# No. 3  contribution rate: 0.0867181502113
# 
# ... ...
# 
# No. 88  contribution rate: 0.000521767479293
# 
# No. 89  contribution rate: 0.000508699929438
# 
# No. 90  contribution rate: 0.000505613673714
# 
# the top  90  contribution rate: 0.99146981703
# 
# Total test samples:49 	avgErrCount:6 	avgErrRate:0.123810
# 
# 
# 
# The result is greatly improved, but still not good enough.
