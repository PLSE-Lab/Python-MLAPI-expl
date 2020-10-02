# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
userMeans = pd.read_csv('../input/meanAndSquareRatings.csv')
userMovie = pd.read_csv('../input/userMovieMat.csv')

meanRatings = list(userMeans['meanRating'])

squareRatings = list(userMeans['squareRating'])

movieIds = [str(i) for i in range(3953)]

userCorrelation = [ [0 for i in range(501)] for j in range(501) ]

#'''
'''

'''
for i in range(21, 41):  # 6041):
    for k in range(1, 501):  # 6041):
        corr_ik = 0.0
        for j in movieIds:
            if i < k:
                iRate = userMovie[j][i]
                kRate = userMovie[j][k]
                if iRate != 0 and kRate != 0:
                    corr_ik += (iRate*kRate)
                    corr_ik /= (squareRatings[i-1]*1.0)**0.5
                    corr_ik /= (squareRatings[k-1]*1.0)**0.5
                    #print(corr_ik)
        userCorrelation[i][k] = corr_ik
        userCorrelation[k][i] = corr_ik
        print(i,k, corr_ik)

#'''

print(userCorrelation[1])

df = pd.DataFrame(userCorrelation)

df.to_csv('vectorCosCorrelation21to40.csv')
