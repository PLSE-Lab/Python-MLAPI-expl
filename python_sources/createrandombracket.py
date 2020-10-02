import pandas as pd
import numpy as np
import re

i_year = 2019
b_random = True

s_kaggle_path = '/Users/scott/datasets/march_madness/kaggle2019/'

s_output = s_kaggle_path + '/Brackets/WinsXGB_LF/expected1.html'

s_file = 'sstrong88_mmadness_submission_20190319_WinsXGB.csv'

s_seeds = 'NCAATourneySeeds.csv'

df_preds = pd.read_csv(s_kaggle_path + s_file)
df_seeds = pd.read_csv(s_kaggle_path + 'NCAATourneySeeds.csv')
df_teams = pd.read_csv(s_kaggle_path + 'Teams.csv')

df_teams = df_teams[df_teams['TeamID'].isin(df_seeds['TeamID'])]

d_id_team = dict(zip(df_teams['TeamID'].values, df_teams['TeamName']))

df_seeds = df_seeds[df_seeds['Season'] == i_year]

df_preds['Year'] = df_preds['ID'].str.split('_', expand=True).values[:, 0].astype(int)
df_preds['Team1'] = df_preds['ID'].str.split('_', expand=True).values[:, 1].astype(int)
df_preds['Team2'] = df_preds['ID'].str.split('_', expand=True).values[:, 2].astype(int)
ls_preds_idx = list(df_preds['ID'])

df_bracket = pd.DataFrame(df_seeds['TeamID'].values, index=df_seeds['Seed'], columns=['TeamID'])
df_bracket['TeamName'] = df_bracket['TeamID'].apply(lambda x: d_id_team[x])
df_bracket['Seed'] = df_seeds['Seed'].values
df_bracket['SeedNum'] = df_bracket['Seed'].apply(lambda x: int(re.findall('\d+', x)[0]))
df_bracket['Region'] = df_bracket['Seed'].apply(lambda x: x[0])

li_sort_order = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
d_sort_order = dict(zip(li_sort_order, range(16)))

df_bracket['sort_idx'] = df_bracket['SeedNum'].apply(lambda x: d_sort_order[x])
df_bracket = df_bracket.sort_values(['Region', 'sort_idx'])
li_teams = list(df_bracket['TeamID'].values)

ls_rounds = ['Playin', 'Round of 64', 'Round of 32', 'Sweet 16', 'Elite 8', 'Final 4', 'Championship', 'Champion']

for i in range(len(ls_rounds) - 1):
    if i == 0:
        # Determine who is involved
        na_playins = np.where(df_bracket['SeedNum'].diff() == 0)[0]
        df_bracket[ls_rounds[i]] = np.nan
        li_remaining = sorted(list(na_playins) + list(na_playins - 1))
        df_bracket[ls_rounds[i]].iloc[li_remaining] = 1
        df_bracket[ls_rounds[i + 1]] = 1

    else:
        li_remaining = list(np.where(df_bracket[ls_rounds[i]] == 1)[0])
        df_bracket[ls_rounds[i + 1]] = np.nan

    # Determine who moves on to the next round
    i_curr = 0
    for j in range(int(len(li_remaining)/2)):
        i_t1_idx = li_remaining[i_curr]
        i_t2_idx = li_remaining[i_curr+1]
        i_team1 = li_teams[i_t1_idx]
        i_team2 = li_teams[i_t2_idx]

        if i_team2 < i_team1:
            # Swap Team1 and Team2
            i_idx_save = i_t1_idx
            i_t1_idx = i_t2_idx
            i_t2_idx = i_idx_save

            i_team_save = i_team1
            i_team1 = i_team2
            i_team2 = i_team_save

        s_loc = str(i_year) + '_' + str(i_team1) + '_' + str(i_team2)
        f_prob_t1_win = df_preds['Pred'].iloc[ls_preds_idx.index(s_loc)]
        if b_random:
            f_rand = np.random.rand()
            if f_rand < f_prob_t1_win:
                # Team1 wins
                df_bracket[ls_rounds[i + 1]].iloc[i_t1_idx] = 1
                df_bracket[ls_rounds[i + 1]].iloc[i_t2_idx] = np.nan
            else:
                # Team2 wins
                df_bracket[ls_rounds[i + 1]].iloc[i_t2_idx] = 1
                df_bracket[ls_rounds[i + 1]].iloc[i_t1_idx] = np.nan

        else:
            if f_prob_t1_win >= 0.5:
                # Team1 wins
                df_bracket[ls_rounds[i + 1]].iloc[i_t1_idx] = 1
                df_bracket[ls_rounds[i + 1]].iloc[i_t2_idx] = np.nan
            else:
                # Team2 wins
                df_bracket[ls_rounds[i + 1]].iloc[i_t2_idx] = 1
                df_bracket[ls_rounds[i + 1]].iloc[i_t1_idx] = np.nan

        i_curr += 2

df_bracket.fillna(0, inplace=True)
for s_col in ls_rounds:
    df_bracket[s_col] = df_bracket[s_col].astype(bool) * df_bracket['TeamName']

with open(s_output, 'w') as fh:
    df_bracket[ls_rounds].to_html(fh)

print('All Done!')

