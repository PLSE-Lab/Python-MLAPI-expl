# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
from scipy.special import erf
from scipy.special import erfinv
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# GLOBAL PARAMETERS
# dictionary of tournament results through 60 games.
# dictionary keys are kaggle game id
# dictionary values are dictionaries with team names, team scores, the score difference 
# and the vegas-predicted score difference
TOURNEY_RESULTS_DICT = {
    '2016_1112_1455' : {'scoreDiff': -10, 'team1': 'Arizona', 'team2': 'Wichita St', 'score1': 55, 'score2': 65, 'vegasPredictedScoreDiff': 1.0} ,
    '2016_1114_1235' : {'scoreDiff': -17, 'team1': 'Ark Little Rock', 'team2': 'Iowa St', 'score1': 61, 'score2': 78, 'vegasPredictedScoreDiff': -6.0} ,
    '2016_1114_1345' : {'scoreDiff': 2, 'team1': 'Ark Little Rock', 'team2': 'Purdue', 'score1': 85, 'score2': 83, 'vegasPredictedScoreDiff': -8.0} ,
    '2016_1122_1242' : {'scoreDiff': -26, 'team1': 'Austin Peay', 'team2': 'Kansas', 'score1': 79, 'score2': 105, 'vegasPredictedScoreDiff': -24.5} ,
    '2016_1124_1463' : {'scoreDiff': -4, 'team1': 'Baylor', 'team2': 'Yale', 'score1': 75, 'score2': 79, 'vegasPredictedScoreDiff': 5.5} ,
    '2016_1138_1274' : {'scoreDiff': -7, 'team1': 'Buffalo', 'team2': 'Miami FL', 'score1': 72, 'score2': 79, 'vegasPredictedScoreDiff': -14.5} ,
    '2016_1139_1403' : {'scoreDiff': 10, 'team1': 'Butler', 'team2': 'Texas Tech', 'score1': 71, 'score2': 61, 'vegasPredictedScoreDiff': 3.5} ,
    '2016_1139_1438' : {'scoreDiff': -8, 'team1': 'Butler', 'team2': 'Virginia', 'score1': 69, 'score2': 77, 'vegasPredictedScoreDiff': -8.5} ,
    '2016_1143_1218' : {'scoreDiff': -11, 'team1': 'California', 'team2': 'Hawaii', 'score1': 66, 'score2': 77, 'vegasPredictedScoreDiff': 6.0} ,
    '2016_1151_1231' : {'scoreDiff': -25, 'team1': 'Chattanooga', 'team2': 'Indiana', 'score1': 74, 'score2': 99, 'vegasPredictedScoreDiff': -10.5} ,
    '2016_1153_1386' : {'scoreDiff': -2, 'team1': 'Cincinnati', 'team2': "St Joseph's PA", 'score1': 76, 'score2': 78, 'vegasPredictedScoreDiff': 3.5} ,
    '2016_1160_1163' : {'scoreDiff': -7, 'team1': 'Colorado', 'team2': 'Connecticut', 'score1': 67, 'score2': 74, 'vegasPredictedScoreDiff': -4.0} ,
    '2016_1163_1242' : {'scoreDiff': -12, 'team1': 'Connecticut', 'team2': 'Kansas', 'score1': 61, 'score2': 73, 'vegasPredictedScoreDiff': -7.5} ,
    '2016_1167_1328' : {'scoreDiff': -14, 'team1': 'CS Bakersfield', 'team2': 'Oklahoma', 'score1': 68, 'score2': 82, 'vegasPredictedScoreDiff': -15.5} ,
    '2016_1173_1393' : {'scoreDiff': -19, 'team1': 'Dayton', 'team2': 'Syracuse', 'score1': 51, 'score2': 70, 'vegasPredictedScoreDiff': -1.0} ,
    '2016_1181_1332' : {'scoreDiff': -14, 'team1': 'Duke', 'team2': 'Oregon', 'score1': 68, 'score2': 82, 'vegasPredictedScoreDiff': -3.5} ,
    '2016_1181_1423' : {'scoreDiff': 8, 'team1': 'Duke', 'team2': 'UNC Wilmington', 'score1': 93, 'score2': 85, 'vegasPredictedScoreDiff': 10.0} ,
    '2016_1181_1463' : {'scoreDiff': 7, 'team1': 'Duke', 'team2': 'Yale', 'score1': 71, 'score2': 64, 'vegasPredictedScoreDiff': 6.5} ,
    '2016_1195_1314' : {'scoreDiff': -16, 'team1': 'FL Gulf Coast', 'team2': 'North Carolina', 'score1': 67, 'score2': 83, 'vegasPredictedScoreDiff': -23.5} ,
    '2016_1201_1428' : {'scoreDiff': -11, 'team1': 'Fresno St', 'team2': 'Utah', 'score1': 69, 'score2': 80, 'vegasPredictedScoreDiff': -8.5} ,
    '2016_1211_1371' : {'scoreDiff': 16, 'team1': 'Gonzaga', 'team2': 'Seton Hall', 'score1': 68, 'score2': 52, 'vegasPredictedScoreDiff': 2.0} ,
    '2016_1211_1393' : {'scoreDiff': -3, 'team1': 'Gonzaga', 'team2': 'Syracuse', 'score1': 60, 'score2': 63, 'vegasPredictedScoreDiff': 4.0} ,
    '2016_1211_1428' : {'scoreDiff': 23, 'team1': 'Gonzaga', 'team2': 'Utah', 'score1': 82, 'score2': 59, 'vegasPredictedScoreDiff': 1.0} ,
    '2016_1214_1438' : {'scoreDiff': -36, 'team1': 'Hampton', 'team2': 'Virginia', 'score1': 45, 'score2': 81, 'vegasPredictedScoreDiff': -23.0} ,
    '2016_1218_1268' : {'scoreDiff': -13, 'team1': 'Hawaii', 'team2': 'Maryland', 'score1': 60, 'score2': 73, 'vegasPredictedScoreDiff': -7.5} ,
    '2016_1221_1332' : {'scoreDiff': -39, 'team1': 'Holy Cross', 'team2': 'Oregon', 'score1': 52, 'score2': 91, 'vegasPredictedScoreDiff': -23.0} ,
    '2016_1231_1246' : {'scoreDiff': 6, 'team1': 'Indiana', 'team2': 'Kentucky', 'score1': 73, 'score2': 67, 'vegasPredictedScoreDiff': -4.0} ,
    '2016_1231_1314' : {'scoreDiff': -15, 'team1': 'Indiana', 'team2': 'North Carolina', 'score1': 86, 'score2': 101, 'vegasPredictedScoreDiff': -5.5} ,
    '2016_1233_1235' : {'scoreDiff': -13, 'team1': 'Iona', 'team2': 'Iowa St', 'score1': 81, 'score2': 94, 'vegasPredictedScoreDiff': -6.5} ,
    '2016_1234_1396' : {'scoreDiff': 2, 'team1': 'Iowa', 'team2': 'Temple', 'score1': 72, 'score2': 70, 'vegasPredictedScoreDiff': 7.0} ,
    '2016_1234_1437' : {'scoreDiff': -19, 'team1': 'Iowa', 'team2': 'Villanova', 'score1': 68, 'score2': 87, 'vegasPredictedScoreDiff': -6.5} ,
    '2016_1235_1438' : {'scoreDiff': -13, 'team1': 'Iowa St', 'team2': 'Virginia', 'score1': 71, 'score2': 84, 'vegasPredictedScoreDiff': -6.5} ,
    '2016_1242_1268' : {'scoreDiff': 16, 'team1': 'Kansas', 'team2': 'Maryland', 'score1': 79, 'score2': 63, 'vegasPredictedScoreDiff': 5.5} ,
    '2016_1242_1437' : {'scoreDiff': -5, 'team1': 'Kansas', 'team2': 'Villanova', 'score1': 59, 'score2': 64, 'vegasPredictedScoreDiff': 2.0} ,
    '2016_1246_1392' : {'scoreDiff': 28, 'team1': 'Kentucky', 'team2': 'Stony Brook', 'score1': 85, 'score2': 57, 'vegasPredictedScoreDiff': 13.5} ,
    '2016_1268_1355' : {'scoreDiff': 5, 'team1': 'Maryland', 'team2': 'S Dakota St', 'score1': 79, 'score2': 74, 'vegasPredictedScoreDiff': 9.0} ,
    '2016_1274_1437' : {'scoreDiff': -23, 'team1': 'Miami', 'team2': 'Villanova', 'score1': 69, 'score2': 92, 'vegasPredictedScoreDiff': -4.0} ,
    '2016_1274_1455' : {'scoreDiff': 8, 'team1': 'Miami', 'team2': 'Wichita St', 'score1': 65, 'score2': 57, 'vegasPredictedScoreDiff': -2.0} ,
    '2016_1276_1323' : {'scoreDiff': -7, 'team1': 'Michigan', 'team2': 'Notre Dame', 'score1': 63, 'score2': 70, 'vegasPredictedScoreDiff': -3.0} ,
    '2016_1277_1292' : {'scoreDiff': -9, 'team1': 'Michigan St', 'team2': 'MTSU', 'score1': 81, 'score2': 90, 'vegasPredictedScoreDiff': 16.5} ,
    '2016_1292_1393' : {'scoreDiff': -25, 'team1': 'MTSU', 'team2': 'Syracuse', 'score1': 50, 'score2': 75, 'vegasPredictedScoreDiff': -6.0} ,
    '2016_1314_1323' : {'scoreDiff': 14, 'team1': 'North Carolina', 'team2': 'Notre Dame', 'score1': 88, 'score2': 74, 'vegasPredictedScoreDiff': 9.5} ,
    '2016_1314_1344' : {'scoreDiff': 19, 'team1': 'North Carolina', 'team2': 'Providence', 'score1': 85, 'score2': 66, 'vegasPredictedScoreDiff': 11.5} ,
    '2016_1320_1400' : {'scoreDiff': 3, 'team1': 'Northern Iowa', 'team2': 'Texas', 'score1': 75, 'score2': 72, 'vegasPredictedScoreDiff': -3.0} ,
    '2016_1320_1401' : {'scoreDiff': -4, 'team1': 'Northern Iowa', 'team2': 'Texas A&M', 'score1': 88, 'score2': 92, 'vegasPredictedScoreDiff': -7.0} ,
    '2016_1323_1372' : {'scoreDiff': 1, 'team1': 'Notre Dame', 'team2': 'SF Austin', 'score1': 76, 'score2': 75, 'vegasPredictedScoreDiff': 2.0} ,
    '2016_1323_1458' : {'scoreDiff': 5, 'team1': 'Notre Dame', 'team2': 'Wisconsin', 'score1': 61, 'score2': 56, 'vegasPredictedScoreDiff': 1.5} ,
    '2016_1328_1332' : {'scoreDiff': 12, 'team1': 'Oklahoma', 'team2': 'Oregon', 'score1': 80, 'score2': 68, 'vegasPredictedScoreDiff': -1.0} ,
    '2016_1328_1401' : {'scoreDiff': 14, 'team1': 'Oklahoma', 'team2': 'Texas A&M', 'score1': 77, 'score2': 63, 'vegasPredictedScoreDiff': 2.5} ,
    '2016_1328_1433' : {'scoreDiff': 4, 'team1': 'Oklahoma', 'team2': 'VA Commonwealth', 'score1': 85, 'score2': 81, 'vegasPredictedScoreDiff': 6.5} ,
    '2016_1332_1386' : {'scoreDiff': 5, 'team1': 'Oregon', 'team2': "St Joseph's PA", 'score1': 69, 'score2': 64, 'vegasPredictedScoreDiff': 7.0} ,
    '2016_1333_1433' : {'scoreDiff': -8, 'team1': 'Oregon St', 'team2': 'VA Commonwealth', 'score1': 67, 'score2': 75, 'vegasPredictedScoreDiff': -4.5} ,
    '2016_1338_1458' : {'scoreDiff': -4, 'team1': 'Pittsburgh', 'team2': 'Wisconsin', 'score1': 43, 'score2': 47, 'vegasPredictedScoreDiff': 1.0} ,
    '2016_1344_1425' : {'scoreDiff': 1, 'team1': 'Providence', 'team2': 'USC', 'score1': 70, 'score2': 69, 'vegasPredictedScoreDiff': 2.5} ,
    '2016_1372_1452' : {'scoreDiff': 14, 'team1': 'SF Austin', 'team2': 'West Virginia', 'score1': 70, 'score2': 56, 'vegasPredictedScoreDiff': -7.0} ,
    '2016_1393_1438' : {'scoreDiff': 6, 'team1': 'Syracuse', 'team2': 'Virginia', 'score1': 68, 'score2': 62, 'vegasPredictedScoreDiff': -8.0} ,
    '2016_1401_1453' : {'scoreDiff': 27, 'team1': 'Texas A&M', 'team2': 'WI Green Bay', 'score1': 92, 'score2': 65, 'vegasPredictedScoreDiff': 13.0} ,
    '2016_1421_1437' : {'scoreDiff': -30, 'team1': 'UNC Asheville', 'team2': 'Villanova', 'score1': 56, 'score2': 86, 'vegasPredictedScoreDiff': -18.0} ,
    '2016_1451_1462' : {'scoreDiff': -18, 'team1': 'Weber St', 'team2': 'Xavier', 'score1': 53, 'score2': 71, 'vegasPredictedScoreDiff': -13.5} ,
    '2016_1458_1462' : {'scoreDiff': 3, 'team1': 'Wisconsin', 'team2': 'Xavier', 'score1': 66, 'score2': 63, 'vegasPredictedScoreDiff': -5.0} 
    }
    
#  standard deviation used for converting between score differences and probabilities
#  Empirically determined from data.
#  Argument can be made for any value near 10.0 or 10.5
std_score_diff = 10.5

# constants
sqrt_2 = np.sqrt(2.0)
max_log_loss = 15.0/np.log(10.0)

# Any results you write to the current directory are saved as output.

# cdf of the normal (0,1) distribution
def cdf_standard_normal(x):
    return 0.5*(1.0 + erf(x/sqrt_2))
    
# inverse cdf of the normal (0,1) distribution
def icdf_standard_normal(p):
    return sqrt_2*erfinv(2.0*p - 1.0)

# functions for converting between probabilities of winning 
# and expected score differences
def convertProbabilityOfWinningToPredictedScoreDifference(probability):
    return -std_score_diff*icdf_standard_normal(1.0 - probability)
    
def convertPredictedScoreDifferenceToProbabilityOfWinning(mean_score_diff):
    return cdf_standard_normal(-mean_score_diff/-std_score_diff)
    
def convertProbabilityOfWinningToProbabilityOfCover(pWin, predScoreDiff):
    return 1.0 - cdf_standard_normal(predScoreDiff/std_score_diff + \
                                     icdf_standard_normal(1.0 - pWin) )

def capProbability(probability):
    epsilon = 1.0E-15
    return np.minimum(np.maximum(np.float64(probability), epsilon), np.float64(1.0) - epsilon)
    
# code adapted from bepd50 for loading data
fdir = '../input'

def chomp(f):
    lines = [l.strip() for l in open('{}/{}'.format(fdir, f)).readlines()]
    return [l for l in lines[1:] if len(l)>0]


def get_probabilities():
    data_dict = {}
    fs = glob.glob('{}/predictions/*csv'.format(fdir))

    for f in fs:
        name = f.split('/')[-1].split('.csv')[0]
        tmp = {}
        lines = chomp(f)
        for l in lines:
            gid, p = l.split(',')
            try:
                p = float(p)
            except ValueError:
                gid, p, = p, gid
                p = float(p)
            
            # only keep the line if it's a game that's been played
            if gid in TOURNEY_RESULTS_DICT.keys():
                tourney_result = TOURNEY_RESULTS_DICT[gid]
                win = (tourney_result['scoreDiff'] > 0)
                cover = (tourney_result['scoreDiff'] > tourney_result['vegasPredictedScoreDiff'])
                
                tmp[gid] = {}
                pWin = capProbability(p)
                
                pCover = convertProbabilityOfWinningToProbabilityOfCover(pWin, tourney_result['vegasPredictedScoreDiff'])
                pCover = capProbability(pCover)
                correctWinPred = (win and (pWin > 0.5)) or (not win and (pWin < 0.5))
                correctCoverPred = (cover and (pCover > 0.5)) or (not cover and (pCover < 0.5))

                tmp[gid]['pScoreDiff'] = convertProbabilityOfWinningToPredictedScoreDifference(pWin)
                tmp[gid]['obsScoreDiff'] = tourney_result['scoreDiff']

                tmp[gid]['pWin'] = pWin
                tmp[gid]['pCover'] = pCover
                tmp[gid]['logLossWin'] = -(win*np.log(pWin) + (1.0 - win)*np.log(1.0 - pWin))
                tmp[gid]['logLossCover'] = -(cover*np.log(pCover) + (1.0 - cover)*np.log(1.0 - pCover))
                tmp[gid]['correctCoverPred'] = correctCoverPred
                tmp[gid]['correctWinPred'] = correctWinPred  
                
        data_dict[name] = tmp
    return data_dict

    
# main program

# Vegas Results
logLoss = 0.0
correctWinPred = 0.0

for gid in TOURNEY_RESULTS_DICT.keys():
    win = (TOURNEY_RESULTS_DICT[gid]['scoreDiff'] > 0)
    pWin = convertPredictedScoreDifferenceToProbabilityOfWinning(TOURNEY_RESULTS_DICT[gid]['vegasPredictedScoreDiff'])
    logLoss -= (np.log(pWin) if win else np.log(1.0 - pWin))
    correctWinPred += ( (win and (pWin > 0.5)) or (not win and (pWin < 0.5)))

logLoss /= len(TOURNEY_RESULTS_DICT.keys())
correctWinPred /= len(TOURNEY_RESULTS_DICT.keys())

print('Vegas Results')
print('=============')
print('Kaggle Log Loss = ', logLoss)
print('Win Prediction Accuracy = ', correctWinPred)

# Individual Results
data_dict = get_probabilities()

results = [['name','logLossWin','logLossCover', 'correctWinPred', 'correctCoverPred', 'chi2', 'chi2dof']]
for name in sorted(data_dict.keys()):
    nGames = 0.0
    logLossWin = 0.0
    logLossCover = 0.0
    correctWinPred = 0.0
    correctCoverPred = 0.0
    chi2 = 0.0

    for gid in data_dict[name].keys():
        nGames += 1
        logLossWin += data_dict[name][gid]['logLossWin']
        logLossCover += data_dict[name][gid]['logLossCover']
        correctWinPred += data_dict[name][gid]['correctWinPred']
        correctCoverPred += data_dict[name][gid]['correctCoverPred']
        chi = (data_dict[name][gid]['pScoreDiff'] - data_dict[name][gid]['obsScoreDiff'])/std_score_diff
        chi2 += chi*chi
    logLossWin /= nGames
    logLossCover /= nGames
    correctWinPred /= nGames
    correctCoverPred /= nGames
    chi2dof = chi2/nGames
    results.append([name, logLossWin, logLossCover, correctWinPred, correctCoverPred, chi2, chi2dof])


output = pd.DataFrame.from_records(results)
output.to_csv('kagglersVsVegasAndScoreBasedMetric.csv')
# output.to_csv('kagglersVsVegas.csv', index=False)
print('Finished')



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
