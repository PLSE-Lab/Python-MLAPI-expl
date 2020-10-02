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


import numpy as np
import re
import copy
import glob
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns

fdir = '../input'

teams_to_seed = {1276: 'W11', 1195: 'W16', 1455: 'Y11', 1221: 'Z16'}

# Second round games
secondRoundIds  = [
  "2016_1114_1235",
  "2016_1139_1438",
  "2016_1163_1242",
  "2016_1181_1463",
  "2016_1211_1428",
  "2016_1218_1268",
  "2016_1231_1246",
  "2016_1234_1437",
  "2016_1274_1455",
  "2016_1292_1393",
  "2016_1314_1344",
  "2016_1320_1401",
  "2016_1323_1372",
  "2016_1328_1433",
  "2016_1332_1386",
  "2016_1458_1462"
]

def chomp(f):
    lines = [l.strip() for l in open('{}/{}'.format(fdir, f)).readlines()]
    return [l for l in lines[1:] if len(l)>0]

def get_probabilities():
    data = []
    fs = glob.glob('{}/predictions/*csv'.format(fdir))

    for f in fs:
        name = f.split('/')[-1].split('.csv')[0]
        tmp = []
        lines = chomp(f)
        for l in lines:
            gid, p = l.split(',')
            try:
                p = float(p)
            except ValueError:
                gid, p, = p, gid
                p = float(p)

            tmp.append(tuple([name, gid, p]))
        tmp.sort()
        data.append(tmp)
    return np.array(data, dtype=[('name', 'U64'), ('id', 'U16'),('pred', 'f8')])

def make_data(selection=2):
    teams = {}
    f = 'Teams.csv'
    lines = chomp(f)
    for l in lines:
        id, name = l.split(',')
        id = int(id)
        teams[id] = name
        teams[name] = id

    seeds = {}
    f = 'TourneySeeds.csv'
    lines = chomp(f)
    for l in lines:
        yr, sd, t = l.split(',')
        yr = int(yr)
        t = int(t)
        if not yr in seeds:
            seeds[yr] = {}
        if yr==2016 and t in teams_to_seed:
            sd = teams_to_seed[t]
        seeds[yr][t] = sd
        seeds[yr][sd] = t
    # play-in for 2016

    data = {}
    slots = {}
    f = 'TourneySlots.csv'
    lines = chomp(f)
    for l in lines:
        yr, slot, strongseed, weakseed = l.split(',')
        yr = int(yr)

        if yr != 2016:
            continue

        if not yr in slots:
            slots[yr] = {}
        if not yr in data:
            data[yr] = []

        if selection==1:
            m = re.match('R([1-9]{1})([A-Z]{1})([1-9]{1})', slot)
            if m:
                rnd, region, game = m.groups()
                tmp = {'round': rnd, 'region': region, 'game': game}

                if not re.match('^[A-Z]{1}[0-9]{2}', strongseed):
                    continue

                t1 = seeds[yr][strongseed]
                t2 = seeds[yr][weakseed]
                if t2<t1:
                    t1, t2 = t2, t1
                tmp['t1'] = t1
                tmp['t2'] = t2
                tmp['name1'] = teams[t1]
                tmp['name2'] = teams[t2]

                data[yr].append(tmp)

    data['matchups'] = {}
    for d in data[2016]:
        k = '2016_{}_{}'.format(d['t1'], d['t2'])
        data['matchups'][k] = copy.copy(d)
    return teams, seeds, data

if __name__=='__main__':
    teams, seeds, data = make_data(selection=2)
    all_probs = get_probabilities()

    ks = secondRoundIds
    ks = sorted([k for k in ks])
    X = []

    game_labels = {}
    for k in ks:
        cc = np.where(all_probs['id'] == k)
        X.append(all_probs[cc]['pred'])
        yr, id1, id2 = k.split('_')
        t1 = teams[int(id1)]
        t2 = teams[int(id2)]
        game_labels[k] = '{} - {}'.format(t1, t2)


    X = np.array(X)
    pca = PCA(n_components=X.shape[0])
    pca.fit(X.transpose())

    sns.set_style('white')
    mx = int(X.shape[0]/3.0) + 1
    idxs = np.argsort((pca.components_[0:mx,:]**2).sum(0))[::-1]
    for i in range(X.shape[0]):
        plt.plot(pca.components_[:,idxs[i]] - i , drawstyle='steps-mid', color='k')
    for i, _ in enumerate(ks):
        k = ks[idxs[i]]
        plt.text(-0.1, -i, game_labels[k],
            horizontalalignment='right', fontsize=6)
    plt.xticks([])
    plt.yticks([])
    plt.xlim(-5, X.shape[0])
    plt.ylim(-X.shape[0]-1, 2)
    plt.title('PCA of 2nd round predictions - 2016')
    plt.xlabel('Principal component number')
    plt.savefig('PCA_2nd_2016.png')

