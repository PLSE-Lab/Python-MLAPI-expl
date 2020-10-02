#!/usr/bin/env python
# coding: utf-8

# In this notebook we will create a Python class for a league table that allows us to add game results one at a time to change the standings. As new results come in, or we make predictions for results, we can add these to the table by calling a single method. In this version of the notebook we will create a simple model that can predict the result of games by looking at recent forms of different teams and how that corresponds to future game results. Scroll to the bottom of this notebook to see the final results. For a basic look at how to make overall predictions without looking at each game, [see this notebook](https://www.kaggle.com/cwthompson/epl-19-20-predicted-final-positions).
# 
# Since the restart of the league, this notebook has been used to make predictions on the results of every game. All predictions can be found at the bottom of the notebook, alongside comparisons with a professional pundit.

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


DATA_DIR = '../input/epl-stats-20192020/'
POSITIONS = np.array(range(1, 21))
RANDOM_STATE = 0


# # Get Previous Results
# Firstly we will read the Premier League data for the season so far.

# In[ ]:


game_data = pd.read_csv(DATA_DIR + 'epl2020.csv')


# Then we will create a new dataframe with the results of each game in. In the csv file we have two rows for each game (one for the home team, one for the away team) so we will have to merge these together. We can then extract the important data: the teams and how many goals they scored. We will also create some extra features that we are going to use in our prediction model later, but we won't populate these features just yet (we will do that later as it will be easier).

# In[ ]:


game_data['scored'] = game_data['scored'].apply(lambda x: str(x))
game_data.sort_values(by=['date', 'h_a'], inplace=True)
game_data.reset_index(inplace=True, drop=True)

# Get game data
results = game_data.groupby(by=['Referee.x', 'date']).agg({'teamId' : ','.join, 'scored' : ','.join}).reset_index()
results['teamA'] = results['teamId'].apply(lambda x: x.split(',')[0])
results['teamB'] = results['teamId'].apply(lambda x: x.split(',')[1])
results['scoredA'] = results['scored'].apply(lambda x: x.split(',')[0]).astype('uint16')
results['scoredB'] = results['scored'].apply(lambda x: x.split(',')[1]).astype('uint16')

# Add additional columns
results.sort_values(by='date', inplace=True)
results.reset_index(inplace=True, drop=True)
results['teamAPosition'] = 0
results['teamBPosition'] = 0
results['teamARecentScored'] = 0
results['teamBRecentScored'] = 0
results['teamARecentConceded'] = 0
results['teamBRecentConceded'] = 0
results['teamARecentPoints'] = 0
results['teamBRecentPoints'] = 0
results = results[['date', 'teamA', 'scoredA', 'teamB', 'scoredB', 'teamAPosition', 'teamBPosition', 'teamARecentScored', 'teamBRecentScored', 'teamARecentConceded', 'teamBRecentConceded', 'teamARecentPoints', 'teamBRecentPoints']]

# Display
#results


# # League Table Class
# The following class can be used to create a league table. The constructor takes only the list of teams in the league. Results can then be added to the table by calling the *add_result* method. This method takes the two teams and the goals they scored.

# In[ ]:


class LeagueTable:
    def __init__(self, teams):
        self.teams = list(teams)
        self.table = pd.DataFrame(np.array([teams, [0] * len(self.teams), [0] * len(self.teams), [0] * len(self.teams), [0] * len(self.teams), [0] * len(self.teams), [0] * len(self.teams), [0] * len(self.teams), [0] * len(self.teams)]).T, columns=['Team', 'P', 'W', 'D', 'L', 'F', 'A', 'GD', 'Pts'])
        self.positions = np.array(range(1, len(self.teams) + 1))
        self.sort_table()
        
    def sort_table(self):
        self.table.sort_values(by=['Team'], ascending=True, inplace=True, ignore_index=True)
        self.table.sort_values(by=['Pts', 'GD', 'F'], ascending=False, inplace=True, ignore_index=True)
        self.table.set_index(self.positions, inplace=True)
    
    def show_table(self):
        return self.table.head(len(self.teams))
    
    def add_result(self, team_a, scored_a, team_b, scored_b):
        # Team A
        self.table.loc[self.table['Team'] == team_a, 'P'] += 1
        self.table.loc[self.table['Team'] == team_a, 'W'] += 1 if int(scored_a) > int(scored_b) else 0
        self.table.loc[self.table['Team'] == team_a, 'D'] += 1 if int(scored_a) == int(scored_b) else 0
        self.table.loc[self.table['Team'] == team_a, 'L'] += 1 if int(scored_a) < int(scored_b) else 0
        self.table.loc[self.table['Team'] == team_a, 'F'] += int(scored_a)
        self.table.loc[self.table['Team'] == team_a, 'A'] += int(scored_b)
        self.table.loc[self.table['Team'] == team_a, 'GD'] += int(scored_a) - int(scored_b)
        self.table.loc[self.table['Team'] == team_a, 'Pts'] += 3 if int(scored_a) > int(scored_b) else 1 if int(scored_a) == int(scored_b) else 0
        # Team B
        self.table.loc[self.table['Team'] == team_b, 'P'] += 1
        self.table.loc[self.table['Team'] == team_b, 'W'] += 1 if int(scored_b) > int(scored_a) else 0
        self.table.loc[self.table['Team'] == team_b, 'D'] += 1 if int(scored_b) == int(scored_a) else 0
        self.table.loc[self.table['Team'] == team_b, 'L'] += 1 if int(scored_b) < int(scored_a) else 0
        self.table.loc[self.table['Team'] == team_b, 'F'] += int(scored_b)
        self.table.loc[self.table['Team'] == team_b, 'A'] += int(scored_a)
        self.table.loc[self.table['Team'] == team_b, 'GD'] += int(scored_b) - int(scored_a)
        self.table.loc[self.table['Team'] == team_b, 'Pts'] += 3 if int(scored_b) > int(scored_a) else 1 if int(scored_b) == int(scored_a) else 0
        # Reorder table
        self.sort_table()


# # Creating The Table
# The table can be created by calling the constructor, as shown below.

# In[ ]:


table = LeagueTable(results['teamA'].unique())
table.show_table()


# We can then add the results that we extracted before to the table. This is done below by iterating through each result and adding it individually. Since we are iterating through each game one at a time, we can also calculate those features that we created earlier. Note that this method isn't perfect as league positions may change between two games, despite them happening at the same time - so this could be improved.

# In[ ]:


for index, row in results.iterrows():
    # Update features
    previous_games = game_data[:index*2]
    results.iloc[index, 5] = table.show_table()[table.show_table()['Team'] == row['teamA']].index[0].astype('uint16')
    results.iloc[index, 6] = table.show_table()[table.show_table()['Team'] == row['teamB']].index[0].astype('uint16')
    results.iloc[index, 7] = previous_games[previous_games['teamId'] == row['teamA']][-5:]['scored'].astype('uint16').sum()
    results.iloc[index, 8] = previous_games[previous_games['teamId'] == row['teamB']][-5:]['scored'].astype('uint16').sum()
    results.iloc[index, 9] = previous_games[previous_games['teamId'] == row['teamA']][-5:]['missed'].astype('uint16').sum()
    results.iloc[index, 10] = previous_games[previous_games['teamId'] == row['teamB']][-5:]['missed'].astype('uint16').sum()
    results.iloc[index, 11] = 3*previous_games[previous_games['teamId'] == row['teamA']][-5:]['wins'].astype('uint16').sum() + previous_games[previous_games['teamId'] == row['teamA']][-5:]['draws'].astype('uint16').sum()
    results.iloc[index, 12] = 3*previous_games[previous_games['teamId'] == row['teamB']][-5:]['wins'].astype('uint16').sum() + previous_games[previous_games['teamId'] == row['teamB']][-5:]['draws'].astype('uint16').sum()
    # Add result to table
    table.add_result(row['teamA'], row['scoredA'], row['teamB'], row['scoredB'])


# The dataset does not contain results of games that happened in Project Restart (since the restarting of the Premier League). So we will need to add the results of these games manually.

# In[ ]:


happened_games = [('Aston Villa', 'Sheffield United', 0, 0), ('Man City', 'Arsenal', 3, 0),
                  ('Aston Villa', 'Chelsea', 1, 2), ('Bournemouth', 'Crystal Palace', 0, 2), ('Brighton', 'Arsenal', 2, 1), ('Everton', 'Liverpool', 0, 0), ('Man City', 'Burnley', 5, 0), ('Newcastle United', 'Sheffield United', 3, 0), ('Norwich', 'Southampton', 0, 3), ('Tottenham', 'Man Utd', 1, 1), ('Watford', 'Leicester', 1, 1), ('West Ham', 'Wolves', 0, 2),
                  ('Burnley', 'Watford', 1, 0), ('Chelsea', 'Man City', 2, 1), ('Leicester', 'Brighton', 0, 0), ('Liverpool', 'Crystal Palace', 4, 0), ('Man Utd', 'Sheffield United', 3, 0), ('Newcastle United', 'Aston Villa', 1, 1), ('Norwich', 'Everton', 0, 1), ('Southampton', 'Arsenal', 0, 2), ('Tottenham', 'West Ham', 2, 0), ('Wolves', 'Bournemouth', 1, 0),
                  ('Aston Villa', 'Wolves', 0, 1), ('Watford', 'Southampton', 1, 3), ('Crystal Palace', 'Burnley', 0, 1), ('Brighton', 'Man Utd', 0, 3), ('Arsenal', 'Norwich', 4, 0), ('Bournemouth', 'Newcastle United', 1, 4), ('Everton', 'Leicester', 2, 1), ('West Ham', 'Chelsea', 3, 2), ('Man City', 'Liverpool', 4, 0), ('Sheffield United', 'Tottenham', 3, 1),
                  ('Burnley', 'Sheffield United', 1, 1), ('Chelsea', 'Watford', 3, 0), ('Leicester', 'Crystal Palace', 3, 0), ('Liverpool', 'Aston Villa', 2, 0), ('Man Utd', 'Bournemouth', 5, 2), ('Newcastle United', 'West Ham', 2, 2), ('Norwich', 'Brighton', 0, 1), ('Southampton', 'Man City', 1, 0), ('Tottenham', 'Everton', 1, 0), ('Wolves', 'Arsenal', 0, 2),
                  ('Arsenal', 'Leicester', 1, 1), ('Aston Villa', 'Man Utd', 0, 3), ('Bournemouth', 'Tottenham', 0, 0), ('Brighton', 'Liverpool', 1, 3), ('Crystal Palace', 'Chelsea', 2, 3), ('Everton', 'Southampton', 1, 1), ('Man City', 'Newcastle United', 5, 0), ('Sheffield United', 'Wolves', 1, 0), ('Watford', 'Norwich', 2, 1), ('West Ham', 'Burnley', 0, 1),
                  ('Aston Villa', 'Crystal Palace', 2, 0), ('Bournemouth', 'Leicester', 4, 1), ('Brighton', 'Man City', 0, 5), ('Liverpool', 'Burnley', 1, 1), ('Man Utd', 'Southampton', 2, 2), ('Norwich', 'West Ham', 0, 4), ('Sheffield United', 'Chelsea', 3, 0), ('Tottenham', 'Arsenal', 2, 1), ('Watford', 'Newcastle United', 2, 1), ('Wolves', 'Everton', 3, 0),
                  ('Arsenal', 'Liverpool', 2, 1), ('Burnley', 'Wolves', 1, 1), ('Chelsea', 'Norwich', 1, 0), ('Crystal Palace', 'Man Utd', 0, 2), ('Everton', 'Aston Villa', 1, 1), ('Leicester', 'Sheffield United', 2, 0), ('Man City', 'Bournemouth', 2, 1), ('Newcastle United', 'Tottenham', 1, 3), ('Southampton', 'Brighton', 1, 1), ('West Ham', 'Watford', 3, 1)]
for game in happened_games:
    # Get win/draw/loss
    home_win = 1 if round(game[2]) > round(game[3]) else 0
    home_draw = 1 if round(game[2]) == round(game[3]) else 0
    home_loss = 1 if round(game[2]) < round(game[3]) else 0
    
    table.add_result(game[0], game[2], game[1], game[3])
    new_row_a = pd.DataFrame([['', '', '', '', '', '', '', '', game[2], game[3], '', '', '', home_win, home_draw, home_loss, '', '', game[0], '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']], columns=game_data.columns)
    new_row_b = pd.DataFrame([['', '', '', '', '', '', '', '', game[3], game[2], '', '', '', home_loss, home_draw, home_win, '', '', game[1], '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']], columns=game_data.columns)
    game_data = pd.concat([game_data, new_row_a, new_row_b], ignore_index=True)


# In[ ]:


table.show_table()


# # Get Remaining Games
# Now we will extract the remaining games. If the order of the games was not important then these could be calculated by looking at which games are possible (every home and away combination), and then looking at which games have already been played. However, our model will use data related to recent form, so we need to know the order in which games are played. For this reason, we are inputting games manually in the correct order.

# In[ ]:


games_left = [('Norwich', 'Burnley'), ('Bournemouth', 'Southampton'), ('Tottenham', 'Leicester'), ('Brighton', 'Newcastle United'), ('Sheffield United', 'Everton'), ('Wolves', 'Crystal Palace'), ('Watford', 'Man City'), ('Aston Villa', 'Arsenal'), ('Man Utd', 'West Ham'), ('Liverpool', 'Chelsea'),
              ('Arsenal', 'Watford'), ('Burnley', 'Brighton'), ('Chelsea', 'Wolves'), ('Crystal Palace', 'Tottenham'), ('Everton', 'Bournemouth'), ('Leicester', 'Man Utd'), ('Man City', 'Norwich'), ('Newcastle United', 'Liverpool'), ('Southampton', 'Sheffield United'), ('West Ham', 'Aston Villa')]


games_left = pd.DataFrame(games_left, columns=['teamA', 'teamB'])
games_left.sample(5, random_state=RANDOM_STATE)


# # Predict Results and Final Table
# Once we know what games still need to be played, we will make predictions for them. Our league table class needs home and away goals to add a result, so we need to be able to predict both of these with our model. To do this, we are going to use sklearn's MultiOutputRegressor along with a KNeighborsRegressor. We are using regression as we are predicting the number of goals scored by both teams.

# In[ ]:


from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


# We will get our training dataset. We will train our model on the features that we extracted earlier: the position of each team in the league, the number of goals scored by each team in the past 5 games, and the number of goals conceded by each team in the past 5 games. The outputs (our target variables) will be the number of goals scored by each team.

# In[ ]:


X_train = results[['teamA', 'teamB', 'teamAPosition', 'teamBPosition', 'teamARecentScored', 'teamBRecentScored', 'teamARecentConceded', 'teamBRecentConceded', 'teamARecentPoints', 'teamBRecentPoints']]
X_train.fillna(0, inplace=True)
Y_train = results[['scoredA', 'scoredB']]


# We will also create some extra features which are based upon historic team performance. The first of these features will indicate whether the team has won the Premier League in the last five seasons. Similarly, the second feature will indicate whether the team has finished in the top 4 in the last five seasons (this feature consists of the traditional top 6 teams and Leicester). The third feature indicates that a team has played in the Championship (the second division) in the last five seasons.

# In[ ]:


# Returns 1 if the team won the Premier League in the previous five seasons
def is_previous_winner(team):
    winners = ['Man City', 'Chelsea', 'Leicester']
    return 1 if team in winners else 0

# Returns 1 if the team finished in the top 4 of the Premier League in the previous five seasons
def is_previous_top_4(team):
    top_4 = ['Man City', 'Liverpool', 'Chelsea', 'Tottenham', 'Man Utd', 'Leicester', 'Arsenal']
    return 1 if team in top_4 else 0

# Returns 1 if the team played at least one season in the Championship (second division) in the previous five seasons
def is_championship(team):
    championships = ['Bournemouth', 'Watford', 'Norwich', 'Burnley', 'Newcastle United', 'Brighton', 'Wolves', 'Sheffield United', 'Aston Villa']
    return 1 if team in championships else 0

X_train['teamAPrevWinner'] = X_train['teamA'].apply(lambda x: is_previous_winner(x))
X_train['teamBPrevWinner'] = X_train['teamB'].apply(lambda x: is_previous_winner(x))
X_train['teamAPrevTop4'] = X_train['teamA'].apply(lambda x: is_previous_top_4(x))
X_train['teamBPrevTop4'] = X_train['teamB'].apply(lambda x: is_previous_top_4(x))
X_train['teamAPrevChampionship'] = X_train['teamA'].apply(lambda x: is_championship(x))
X_train['teamBPrevChampionship'] = X_train['teamB'].apply(lambda x: is_championship(x))

X_train.drop(['teamA', 'teamB'], inplace=True, axis=1)


# So that all of our features have a similar impact on the model we will use the StandardScaler to standardise them.

# In[ ]:


scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_train[:,0:2] *= 2.5
X_train[:,2:8] *= 1.5


# We will then create the model and fit it with our training data.

# In[ ]:


model = MultiOutputRegressor(KNeighborsRegressor(n_neighbors=3, weights='distance')).fit(X_train, Y_train)


# Now we have our model. We can use this to predict results for every remaining game, and then add these onto our table. Since our model makes use of the recent form of each team we must update this after each prediction.

# In[ ]:


for index, row in games_left.iterrows():
    # Get recent form features
    a_position = table.show_table()[table.show_table()['Team'] == row['teamA']].index[0].astype('uint16')
    b_position = table.show_table()[table.show_table()['Team'] == row['teamB']].index[0].astype('uint16')
    a_scored = game_data[game_data['teamId'] == row['teamA']][-5:]['scored'].astype('uint16').sum()
    b_scored = game_data[game_data['teamId'] == row['teamB']][-5:]['scored'].astype('uint16').sum()
    a_conceded = game_data[game_data['teamId'] == row['teamA']][-5:]['missed'].astype('uint16').sum()
    b_conceded = game_data[game_data['teamId'] == row['teamB']][-5:]['missed'].astype('uint16').sum()
    a_points = 3*game_data[game_data['teamId'] == row['teamA']][-5:]['wins'].astype('uint16').sum() + game_data[game_data['teamId'] == row['teamA']][-5:]['draws'].astype('uint16').sum()
    b_points = 3*game_data[game_data['teamId'] == row['teamB']][-5:]['wins'].astype('uint16').sum() + game_data[game_data['teamId'] == row['teamB']][-5:]['draws'].astype('uint16').sum()
    
    
    # Get extra features
    a_prev = is_previous_winner(row['teamA'])
    b_prev = is_previous_winner(row['teamB'])
    a_top4 = is_previous_top_4(row['teamA'])
    b_top4 = is_previous_top_4(row['teamB'])
    a_cham = is_championship(row['teamA'])
    b_cham = is_championship(row['teamB'])
    
    # Make game prediction
    X_pred = np.array([a_position, b_position, a_scored, b_scored, a_conceded, b_conceded, a_points, b_points, a_prev, b_prev, a_top4, b_top4, a_cham, b_cham]).reshape(1, -1)
    X_pred = scaler.transform(X_pred)
    X_pred[:,0:2] *= 2.5
    X_pred[:,2:8] *= 1.5
    goals = model.predict(X_pred)
    
    # Add result to the table
    table.add_result(row['teamA'], round(goals[0][0]), row['teamB'], round(goals[0][1]))
    
    # Get win/draw/loss
    home_win = 1 if round(goals[0][0]) > round(goals[0][1]) else 0
    home_draw = 1 if round(goals[0][0]) == round(goals[0][1]) else 0
    home_loss = 1 if round(goals[0][0]) < round(goals[0][1]) else 0
    
    # Save the result (updating recent form)
    new_row_a = pd.DataFrame([['', '', '', '', '', '', '', '', goals[0][0], goals[0][1], '', '', '', home_win, home_draw, home_loss, '', '', row['teamA'], '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']], columns=game_data.columns)
    new_row_b = pd.DataFrame([['', '', '', '', '', '', '', '', goals[0][1], goals[0][0], '', '', '', home_loss, home_draw, home_win, '', '', row['teamB'], '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']], columns=game_data.columns)
    game_data = pd.concat([game_data, new_row_a, new_row_b], ignore_index=True)

# Show the final table
table.show_table()


# # Discussion
# I have temporarily stopped updating the discussion section of this notebook while I do weekly predictions for the Premier League, however you can read an [old discussion page here](https://www.kaggle.com/cwthompson/epl-19-20-game-by-game-predictions?scriptVersionId=34792349#Discussion). That discussion page read the following:
# 
# *Our model has predicted that Liverpool (unsurprisingly) will win the league by winning all of their remaining games. Man City take second place by winning all but two of their remaining games (one draw and one loss) while Leicester hang onto third without losing any remaining games. Chelsea hang onto fourth place with a close battle against Man United, only two points separate them in this model. Wolves and Sheffield United take the two remaining Europa League spot.*
# 
# *Our model predicts that the relegation battle will be incredibly tight. Norwich manage to scrape 10 points in their remaining 9 games which takes them 2 points above Aston Villa but not enough to survive. Watford (+5 points), Bournemouth(+5 points), and Brighton(+3 points) are all predicted to finish on 32 points, but it will be Watford who get relegated with the lowest goal difference - actually only one lower than Bournemouth.*
# 
# *Our model for predicting games is incredibly simplistic. It only takes into account the teams that are playing, their position after 29 games (or 28 games for teams with a game in hand), and results against similarly positioned teams. To create a more accurate model we could modify this so that the training data uses team positions at that point in the season, and we could also take into account recent form (for example Bournemouth have a much better recent form than Watford and Brighton). These are things that could be explored in future versions of the notebook.*

# # On-Going Prediction Evaluation
# 
# The models created in this notebook are being used to make predictions on all remaining games in the Premier League since the restart of the league. The predictions and true results so far can be found below.
# 
# |Game Week | Home Team | Away Team | True Score | Predicted Score | Correct Result? | Notebook Version
# | --- | --- | --- | --- | --- | --- | --- |
# | 29 | Aston Villa | Sheffield United | 0 - 0 | 0 - 3 | N | 3 |
# | 29 | Man City | Arsenal | 3 - 0 | 1 - 1 | N | 3 |
# | 30 | Norwich | Southampton | 0 - 3 | 1 - 1 | N | 4 |
# | 30 | Tottenham | Man Utd | 1 - 1 | 1 - 2 | N | 4 |
# | 30 | Watford | Leicester | 1 - 1 | 1 - 2 | N | 4 |
# | 30 | Brighton | Arsenal | 2 - 1 | 1 - 2 | N | 4 |
# | 30 | West Ham | Wolves | 0 - 2 | 0 - 2 | Y* | 4 |
# | 30 | Bournemouth | Crystal Palace | 0 - 2 | 1 - 1 | N | 4 |
# | 30 | Newcastle United | Sheffield United | 3 - 0 | 1 - 1 | N | 4 |
# | 30 | Aston Villa | Chelsea | 1 - 2 | 0 - 3 | Y | 4 |
# | 30 | Everton | Liverpool | 0 - 0 | 1 - 3 | N | 4 |
# | 30 | Man City | Burnley | 5 - 0 | 2 - 1 | Y | 4 |
# | 31 | Leicester | Brighton | 0 - 0 | 1 - 1 | Y | 5 |
# | 31 | Tottenham | West Ham | 2 - 0 | 1 - 1 | N | 5 |
# | 31 | Man Utd | Sheffield United | 3 - 0 | 2 - 0 | Y | 5 |
# | 31 | Newcastle United | Aston Villa | 1 - 1 | 1 - 1 | Y* | 5 |
# | 31 | Norwich | Everton | 0 - 1 | 2 - 1 | N | 5 |
# | 31 | Wolves | Bournemouth | 1 - 0 | 0 - 1 | N | 5 |
# | 31 | Liverpool | Crystal Palace | 4 - 0 | 0 - 2 | N | 5 |
# | 31 | Burnley | Watford | 1 - 0 | 2 - 1 | Y | 5 |
# | 31 | Southampton | Arsenal | 0 - 2 | 1 - 2 | Y | 5 |
# | 31 | Chelsea | Man City | 2 - 1 | 1 - 3 | N | 5 |
# | 32 | Aston Villa | Wolves | 0 - 1 | 1 - 1 | N | 6 |
# | 32 | Watford | Southampton | 1 - 3 | 1 - 0 | N | 6 |
# | 32 | Crystal Palace | Burnley | 0 - 1 | 1 - 1 | N | 6 |
# | 32 | Brighton | Man Utd | 0 - 3 | 1 - 2 | Y | 6 |
# | 32 | Arsenal | Norwich | 4 - 0 | 1 - 2 | N | 6 |
# | 32 | Bournemouth | Newcastle United | 1 - 4 | 0 - 2 | Y | 6 |
# | 32 | Everton | Leicester | 2 - 1 | 1 - 2 | N | 6 |
# | 32 | West Ham | Chelsea | 3 - 2 | 1 - 1 | N | 6 |
# | 32 | Sheffield United | Tottenham | 3 - 1 | 1 - 1 | N | 6 |
# | 32 | Man City | Liverpool | 4 - 0 | 1 - 2 | N | 6 |
# | 33 | Chelsea | Watford | 3 - 0 | 4 - 2 | Y | 7 |
# | 33 | Wolves | Arsenal | 0 - 2 | 2 - 3 | Y | 7 |
# | 33 | Man Utd | Bournemouth | 5 - 2 | 1 - 3 | N | 7 |
# | 33 | Leicester | Crystal Palace | 3 - 0 | 1 - 1 | N | 7 |
# | 33 | Norwich | Brighton | 0 - 1 | 1 - 1 | N | 7 |
# | 33 | Burnley | Sheffield United | 1 - 1 | 1 - 1 | Y* | 7 |
# | 33 | Newcastle United | West Ham | 2 - 2 | 0 - 1 | N | 7 |
# | 33 | Liverpool | Aston Villa | 2 - 0 | 1 - 2 | N | 7 |
# | 33 | Southampton | Man City | 1 - 0 | 1 - 1 | N | 7 |
# | 33 | Tottenham | Everton | 1 - 0 | 0 - 1 | N | 7 |
# | 34 | Crystal Palace | Chelsea | 2 - 3 | 0 - 1 | Y | 8 |
# | 34 | Watford | Norwich | 2 - 1 | 1 - 1 | N | 8 |
# | 34 | Arsenal | Leicester | 1 - 1 | 1 - 2 | N | 8 |
# | 34 | Man City | Newcastle United | 5 - 0 | 3 - 2 | Y | 8 |
# | 34 | Sheffield United | Wolves | 1 - 0 | 1 - 2 | N | 8 |
# | 34 | West Ham | Burnley | 0 - 1 | 0 - 1 | Y* | 8 |
# | 34 | Brighton | Liverpool | 1 - 3| 1 - 2 | Y | 8 |
# | 34 | Bournemouth | Tottenham | 0 - 0 | 0 - 3 | N | 8 |
# | 34 | Everton | Southampton | 1 - 1 | 1 - 1 | Y* | 8 |
# | 34 | Aston Villa | Man Utd | 0 - 3 | 1 - 2 | Y | 8 |
# | 35 | Norwich | West Ham | 0 - 4 | 0 - 2 | Y | 9 |
# | 35 | Watford | Newcastle United | 2 - 1 | 1 - 2 | N | 9 |
# | 35 | Liverpool | Burnley | 1 - 1 | 1 - 0 | N | 9 |
# | 35 | Sheffield United | Chelsea | 3 - 0 | 0 - 2 | N | 9 |
# | 35 | Brighton | Man City | 0 - 5 | 1 - 1 | N | 9 |
# | 35 | Wolves | Everton | 3 - 0 | 1 - 1 | N | 9 |
# | 35 | Aston Villa | Crystal Palace | 2 - 0 | 0 - 3 | N | 9 |
# | 35 | Tottenham | Arsenal | 2 - 1 | 3 - 2 | Y | 9 |
# | 35 | Bournemouth | Leicester | 4 - 1 | 0 - 2 | N | 9 |
# | 35 | Man Utd | Southampton | 2 - 2 | 2 - 1 | N | 9 |
# | 36 | Chelsea | Norwich | 1 - 0 | 1 - 2 | N | 10 |
# | 36 | Burnley | Wolves | 1 - 1 | 1 - 1 | Y* | 10 |
# | 36 | Man City | Bournemouth | 2 - 1 | 4 - 2 | Y | 10 |
# | 36 | Newcastle United | Tottenham | 1 - 3 | 0 - 2 | Y | 10 |
# | 36 | Arsenal | Liverpool | 2 - 1 | 1 - 2 | N | 10 |
# | 36 | Everton | Aston Villa | 1 - 1 | 0 - 0 | Y | 10 |
# | 36 | Leicester | Sheffield United | 2 - 0 | 1 - 0 | Y | 10 |
# | 36 | Crystal Palace | Man Utd | 0 - 2 | 2 - 2 | N | 10 |
# | 36 | Southampton | Brighton | 1 - 1 | 1 - 1 | Y* | 10 |
# | 36 | West Ham | Watford | 3 - 1 | 3 - 1 | Y* | 10 |
# | 37 | Norwich | Burnley | x - x | 0 - 2 | x | 11 |
# | 37 | Bournemouth | Southampton | x - x | 0 - 1 | x | 11 |
# | 37 | Tottenham | Leicester | x - x | 1 - 2 | x | 11 |
# | 37 | Brighton | Newcastle United | x - x | 0 - 2 | x | 11 |
# | 37 | Sheffield United | Everton | x - x | 1 - 1 | x | 11 |
# | 37 | Wolves | Crystal Palace | x - x | 1 - 1 | x | 11 |
# | 37 | Watford | Man City | x - x | 1 - 1 | x | 11 |
# | 37 | Aston Villa | Arsenal | x - x | 1 - 3 | x | 11 |
# | 37 | Man Utd | West Ham | x - x | 2 - 0 | x | 11 |
# | 37 | Liverpool | Chelsea | x - x | 1 - 2 | x | 11 |
# 
# <br>
# 
# Comparisons can also be made against the predictions on a BBC professional pundit, Mark Lawrenson. Lawro makes a prediction for every Premier League match alongside a different celebrity for each set of games. 10 points are awarded for a correctly predicted results (W, D, L), while 40 points are awarded for a correctly predicted score. As a method of evaluating the model in this notebook, the predictions of the model will be compared to Lawro's predictions.  
# For reference; Lawro's predictions from weeks [29](https://www.bbc.co.uk/sport/football/53049710), [30](https://www.bbc.co.uk/sport/football/53094387), [31](https://www.bbc.co.uk/sport/football/53116030), [32a](https://www.bbc.co.uk/sport/football/53173025), [32b](https://www.bbc.co.uk/sport/football/53219599), [33](https://www.bbc.co.uk/sport/football/53258585), [34](https://www.bbc.co.uk/sport/football/53308609), [35](https://www.bbc.co.uk/sport/football/53341826), [36](https://www.bbc.co.uk/sport/football/53389016), [37a](https://www.bbc.co.uk/sport/football/53431016)...
# 
# The table below shows a weekly comparison of prediction points.
# 
# | Week | Model Points | Lawro Points |
# | --- | --- | --- |
# | 29 | 0 | 20 |
# | 30 | 60 | 60 |
# | 31 | 80 | 160 |
# | 32 | 20 | 30 |
# | 33 | 60 | 70 |
# | 34 | 120 | 80 |
# | 35 | 20 | 40 |
# | 36 | 160 | 80 |
