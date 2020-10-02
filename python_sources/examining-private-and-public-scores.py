import numpy as np
import pandas as pd
from ggplot import *

dataComp = pd.read_csv("../input/Competitions.csv")
dataSubm = pd.read_csv("../input/Submissions.csv")
dataEval = pd.read_csv("../input/EvaluationAlgorithms.csv")
dataTeams = pd.read_csv("../input/Teams.csv")
dataUsers = pd.read_csv("../input/Users.csv")

dataComp.DateEnabled = dataComp.DateEnabled.apply(lambda x: pd.to_datetime(x[0:10]))
dataComp.Deadline = dataComp.Deadline.apply(lambda x: pd.to_datetime(x[0:10]))

EvalAlgs = (dataComp.groupby('EvaluationAlgorithmId').apply(lambda x: len(x)))
EvalAlgs.sort(ascending = 0)
print (EvalAlgs.head())
print (EvalAlgs.sum())

print (dataEval[dataEval.Id == 5])
print (dataEval[dataEval.Id == 2])
print (dataEval[dataEval.Id == 14])

def findMaxScores(IDList, teamNumber):
    beta = pd.DataFrame(data = [], columns = dataSubm.columns.values.tolist() + ['CompetitionId'])
    for compID in IDList:
        curCompTeams = dataTeams[dataTeams.CompetitionId == compID][dataTeams.Ranking <= teamNumber]
        for teamID in curCompTeams.Id:
            N = dataSubm[dataSubm.TeamId == teamID]
            M = N.PrivateScore.dropna().apply(lambda x: float(x)).idxmax()
            K = dataSubm[dataSubm.Id == dataSubm.Id[M]]
            K['CompetitionId'] = int(compID)
            beta = beta.append(K)
    
    return beta
    
def findMinScores(IDList, teamNumber):
    beta = pd.DataFrame(data = [], columns = dataSubm.columns.values.tolist() + ['CompetitionId'])
    for compID in IDList:
        curCompTeams = dataTeams[dataTeams.CompetitionId == compID][dataTeams.Ranking <= teamNumber]
        for teamID in curCompTeams.Id:
            N = dataSubm[dataSubm.TeamId == teamID]
            M = N.PrivateScore.dropna().apply(lambda x: float(x)).idxmin()
            K = dataSubm[dataSubm.Id == dataSubm.Id[M]]
            K['CompetitionId'] = int(compID)
            beta = beta.append(K)
    
    return beta
    
AUCComps = dataComp[dataComp.EvaluationAlgorithmId == 5]
AUCsc = findMaxScores (AUCComps.Id, 5)
AUCsc.PrivateScore = AUCsc.PrivateScore.apply(lambda x: float(x))
AUCsc.PublicScore = AUCsc.PublicScore.apply(lambda x: float(x))
AUCDF = AUCsc[['PrivateScore', 'PublicScore', 'CompetitionId']].copy().groupby('CompetitionId').mean()

print(dataComp[dataComp.Id == 2752])
print(dataComp[dataComp.Id == 3126])
print(dataComp[dataComp.Id == 3507])
print(dataComp[dataComp.Id == 3933])

AUCDF['Label'] = ''
AUCDF.Label[2752] = 'Web-spam diagnostic'
AUCDF.Label[3126] = 'Algorithmic composition'
AUCDF.Label[3507] = 'The ICML 2013 \n Bird Challenge'
AUCDF.Label[3933] = 'MLSP 2014 Schizophrenia Classification'

AUCplot = ggplot(aes(x='PrivateScore',y = 'PublicScore', label = 'Label'), data=AUCDF)
AUCplot = AUCplot + geom_point() + xlim(-0.1,1.1) + ylim(-0.1,1.1) + geom_text(hjust=-0.15, vjust=-0.05)

ggsave (AUCplot, "AUCprivate_public_scores.png")

CAComps = dataComp[dataComp.EvaluationAlgorithmId == 14]
CAsc = findMaxScores (CAComps.Id, 5)
CAsc.PrivateScore = CAsc.PrivateScore.apply(lambda x: float(x))
CAsc.PublicScore = CAsc.PublicScore.apply(lambda x: float(x))
CADF = CAsc[['PrivateScore', 'PublicScore', 'CompetitionId']].copy().groupby('CompetitionId').mean()
print(dataComp[dataComp.Id == 3439])
print(dataComp[dataComp.Id == 3497])
CADF['Label'] = ''
CADF.Label[3439] = 'Handwritten digits recognition'
CADF.Label[3497] = 'Missing and Imbalanced Data'

CAplot = ggplot(aes(x='PrivateScore',y = 'PublicScore', label = 'Label'), data=CADF)
CAplot = CAplot + geom_point() + xlim(-0.1,1.1) + ylim(-0.1,1.1) + geom_text(vjust=-0.05)

ggsave (CAplot, "CAprivate_public_scores.png")

RMSEComps = dataComp[dataComp.EvaluationAlgorithmId == 2]
RMSEsc = findMinScores (RMSEComps.Id, 5)
RMSEsc.PrivateScore = RMSEsc.PrivateScore.apply(lambda x: float(x))
RMSEsc.PublicScore = RMSEsc.PublicScore.apply(lambda x: float(x))
RMSEDF = RMSEsc[['PrivateScore', 'PublicScore', 'CompetitionId']].copy().groupby('CompetitionId').mean()
print(dataComp[dataComp.Id == 4272])
RMSEDF['Label'] = ''
RMSEDF.Label[4272] = 'Restaurant Revenue Prediction'

RMSEplot = ggplot(aes(x='PrivateScore',y = 'PublicScore', label = 'Label'), data=RMSEDF)
RMSEplot = RMSEplot + geom_point() + geom_text(vjust=-0.1)

ggsave (RMSEplot, "RMSEprivate_public_scores.png")

RMSEplot = ggplot(aes(x='PrivateScore',y = 'PublicScore'), data=RMSEDF)
RMSEplot = RMSEplot + geom_point() + xlim(-0.1,1.1) + ylim(-0.1,1.1)

ggsave (RMSEplot, "RMSEprivate_public_scores_trunc.png")








