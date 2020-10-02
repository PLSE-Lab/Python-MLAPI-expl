# THIS SCRIPT IS AWFULL!!!


import pandas as pd
import numpy as np


def split_usd(usd, n):
    # the dataset does include how to distribute money?
    # use this function as an approximation
    if n == 1:
        return np.array([usd])
        
    else:
        alpha = np.arange(n, 0, -1)
        alpha = 1.0 * alpha / np.sum(alpha)
        
        return usd * alpha
    
    
competitions = pd.read_csv('../input/Competitions.csv',)


# 1 means USD(money!!!)
competitions = competitions[competitions['RewardTypeId'] == 1]


# correct ge competitions
# thank you for Jules

# (1) correct RewardQuantity of milestone phase 
# see https://www.kaggle.com/khyh00/meta-kaggle/ge-competitions/run/69431
competitions.loc[competitions['Id']==3521, 'RewardQuantity'] = 30000.0

# (2) remove main phase
competitions = competitions[competitions['Id'] != 3611]

print(competitions.loc[competitions['CompetitionHostSegmentId'] == 7, ['Id', 'Title', 'RewardQuantity']])


teams = pd.read_csv('../input/Teams.csv',)

team_memberships = pd.read_csv('../input/TeamMemberships.csv',)

users = pd.read_csv('../input/Users.csv',)


# user id as key and list for usd
tmp = {}

for (i, c) in competitions.iterrows():
    competition_id = c['Id']
    competition_name = c['CompetitionName']
    
    competition_usd = c['RewardQuantity']
    n_winners =  c['NumPrizes']

    if competition_usd <= 0.0:
        continue
    
    usd_all = split_usd(competition_usd, n_winners)
    
    for j in range(n_winners):
        rank = j + 1
        usd_rank = usd_all[j]
        
        t = teams[(teams['CompetitionId'] == competition_id) & (teams['Ranking'] == rank)]
        
        # FIXME
        if t.shape[0] > 1:
            # tier?
            continue
        
        elif t.shape[0] == 0:
            # no team?
            continue
        
        team_id = t.iloc[0]['Id']
        members = team_memberships[team_memberships['TeamId'] == team_id]
        
        n_members = members.shape[0]
        usd_user = 1.0 * usd_rank / n_members # money is distributed equally?
        
        for (k, m) in members.iterrows():
            user_id = int(m['UserId'])
            if not user_id in tmp:
                tmp[user_id] = []

            tmp[user_id].append(usd_user)
            
            
tmp = np.array([(user_id, np.sum(tmp[user_id])) for user_id in tmp])
users_prize = pd.DataFrame(tmp, columns=['Id', 'Prize'])


# merge and sort
df = pd.merge(users_prize, users, how='inner', on='Id')
df.sort('Prize', ascending=False, inplace=True)
df.reset_index(drop=True, inplace=True)


# output by html
o_head = '''
<!DOCTYPE html>
<html>

<head>
<meta charset="UTF-8"> 

<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/css/bootstrap.min.css">
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/css/bootstrap-theme.min.css">
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/js/bootstrap.min.js"></script>

</head>

<body>

<h3>Who earned the most?</h3>

<ul>
<li><mark>This is not actual money people received, it is just a test to see whether we can estimate it from the given dataset.</mark></li>
<li>The dataset does not provide how to distribute prize money, so a simple rule is applied instead.</li>
<li>It is assumed that prize money is distributed equally among teams.</li>
<li>GE competitions are excluded for now.</li>
<li>Notify me if you are offended by this script.</li>
</ul>

<br />
<br />
<br />

<table class="table table-striped">
<thead>
<tr>
<th>Ranking</th>
<th>User Name</th>
<th>Total Prize (USD)</th>
</tr>
</thead>

<tbody>
'''

o_e = '''
<tr>
<td>{user_rank}</td>
<td><a href="https://www.kaggle.com/users/{user_id}/">{user_name}</a></td>
<td>{user_usd:0.2f}</td>
</tr>
'''


o_foot = '''
</tbody>
</table>

</body>
</html>
'''

o = ''
o += o_head

for (i, u) in df.iterrows():
    o += o_e.format(**{
        'user_rank': i+1,
        'user_id': int(u['Id']),
        'user_name': u['DisplayName'],
        'user_usd': u['Prize'],
        })
        
    # display top 100
    if (i+1) >= 100:
        break
    
o += o_foot

with open("results.html","wb") as outfile:
    outfile.write(o.encode("utf-8"))
    
    
    
    
    
    
    
    
    
    