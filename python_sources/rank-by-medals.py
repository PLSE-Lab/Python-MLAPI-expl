import pandas as pd
import numpy as np

competitions = pd.read_csv('../input/Competitions.csv', index_col='Id')

teams = pd.read_csv('../input/Teams.csv')
team_memberships = pd.read_csv('../input/TeamMemberships.csv', index_col='Id')
users = pd.read_csv('../input/Users.csv', index_col='Id')

gold = []
silver = []
bronze = []

for rank in [1,2,3]:
    rank_team = teams[teams['Ranking'] == rank]
    for (i, t) in rank_team.iterrows():
        c = competitions.loc[t['CompetitionId']]
        if (c['RewardTypeId'] != 8) & (c['HasLeaderboard'] == True):
            members = team_memberships[team_memberships['TeamId'] == t['Id']]
            for (k, m) in members.iterrows():
                user_id = int(m['UserId'])
                if rank==1:
                    gold.append(user_id)
                if rank==2:
                    silver.append(user_id)
                if rank==3:
                    bronze.append(user_id)


index = np.unique(gold + silver + bronze)

results = pd.DataFrame(index = index, columns = ['Gold', 'Silver', 'Bronze'])

for i in index:
    results['Gold'][i] = np.sum(np.array(gold)==i) 
    results['Silver'][i] = np.sum(np.array(silver)==i) 
    results['Bronze'][i] = np.sum(np.array(bronze)==i) 


results['Name'] = users.loc[index]['DisplayName']
results['KaggleRank'] = users.loc[index]['Ranking']
results['KagglePoints'] = users.loc[index]['Points']   
results['MedalPoints'] = results['Gold']*100 + results['Silver']*1 + results['Bronze']*0.01 + results['KagglePoints']*1e-12
results = results.sort('MedalPoints', ascending=False)
results.dropna(inplace=True)
print(results)
results.to_csv('results.csv')

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

<table class="table table-striped">
<thead>
<tr>
<th>Ranking</th>
<th>User Name</th>
<th>Gold</th>
<th>Silver</th>
<th>Bronze</th>
<th>Kaggle Ranking</th>
</tr>
</thead>

<tbody>
'''

o_e = '''
<tr>
<td>{user_rank}</td>
<td><a href="https://www.kaggle.com/users/{user_id}/">{user_name}</a></td>
<td>{user_gold:d}</td>
<td>{user_silver:d}</td>
<td>{user_bronze:d}</td>
<td>{user_kaggle_ranking:d}</td>
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
k = 0
for (i, u) in results.iterrows():
    k += 1
    o += o_e.format(**{
        'user_rank': k,
        'user_id': i,
        'user_name': u['Name'],
        'user_gold': u['Gold'],
        'user_silver': u['Silver'],
        'user_bronze': u['Bronze'],
        'user_kaggle_ranking': int(u['KaggleRank']),
        })
        
    
o += o_foot

with open("results.html","wb") as outfile:
    outfile.write(o.encode("utf-8"))