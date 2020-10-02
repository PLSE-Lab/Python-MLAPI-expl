
import numpy as np 
import pandas as pd 


from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

deliveries = pd.read_csv('../input/deliveries.csv')
matches = pd.read_csv('../input/matches.csv')

def add_batsmen(roster,sample):
    batsmen = sample.groupby('batsman')
    for batsman in batsmen.groups:
        teams = batsmen.get_group(batsman).groupby('batting_team')
        for team in teams.groups:
            roster=roster.append(pd.DataFrame({'type':'batsmen','player':batsman,'team':team, 'match_id':teams.get_group(team)['match_id'].unique()}))    
    return roster

def add_bowlers(roster,sample):
    bowlers = sample.groupby('bowler')
    for bowler in bowlers.groups:
        teams = bowlers.get_group(bowler).groupby('bowling_team')
        for team in teams.groups:
            roster=roster.append(pd.DataFrame({'type':'bowler','player':bowler,'team':team, 'match_id':teams.get_group(team)['match_id'].unique()}))
    return roster
    
def player_roster(size):
    sample = deliveries.head(size)
    roster = pd.DataFrame()
    roster = add_batsmen(roster,sample)
    roster = add_bowlers(roster,sample)
    return roster
    
def important_toss_label(sample):
    results = pd.DataFrame()
    sample.loc[:,'toss_importance']=sample['toss_winner']==sample['winner']
    return sample
    
def close_failed_chases():
    seasons = matches[matches['win_by_runs']!=0].groupby('season')
    results = pd.Series()
    for season in seasons.groups:
        print(season, seasons.get_group(season)['win_by_runs'].mean())
    #plt.scatter(results[0],results[1])
    #plt.show()
    
def find_extra_balls(sample):
    return sample[sample['noball_runs']!=0]['noball_runs'].count()+sample[sample['wide_runs']!=0]['wide_runs'].count()

def find_scoring_shots_with(sample,runs):
    scoring_shot = sample[sample['batsman_runs']==runs]
    return scoring_shot['over']*6+scoring_shot['ball']
    
    #return scoring_shot['over']*6+scoring_shot['ball']

def find_scoring_shots(sample):
    return sample[sample['batsman_runs']!=0]['batsman_runs'].count()
    
def find_dots(sample):
    return sample[sample['total_runs']==0]['total_runs'].count()

def find_boundaries_runs(sample):
    return sample[(sample['batsman_runs']==4) | (sample['batsman_runs']==6)]['batsman_runs'].sum()

def find_byes_legbyes(sample):
    return sample[(sample['bye_runs']!=0) | (sample['legbye_runs']!=0)]['extra_runs'].count()
    
def match_scores(sample):
    matches_played = sample.groupby('match_id')
    for match in matches_played.groups:
        first_total = 0
        second_total = 0
        first_wickets = 0
        second_wickets = 0
        innings = matches_played.get_group(match).groupby('inning')
        if 1 in innings.groups:
            first = innings.get_group(1)
            first_total=first['total_runs'].sum()
            first_wickets = first[first['player_dismissed']!=""]['player_dismissed'].count()
            first_balls = first['over'].count()-find_extra_balls(first)
        if 2 in innings.groups:
            second = innings.get_group(2)
            second_total=second['total_runs'].sum()
            second_wickets = second[second['player_dismissed']!=""]['player_dismissed'].count()
            second_balls = second['over'].count()-find_extra_balls(second)
        matches.loc[matches['id']==match,'first_total'] = first_total 
        matches.loc[matches['id']==match,'first_balls'] = first_balls
        matches.loc[matches['id']==match,'first_wickets'] = first_wickets
        matches.loc[matches['id']==match,'second_total'] = second_total
        matches.loc[matches['id']==match,'second_balls'] = second_balls
        matches.loc[matches['id']==match,'second_wickets'] = second_wickets
    matches.to_csv("augmented_matches.csv")
    return matches
    
def process_inning(match,innings_details, inning, batting_order):
    batsman_performance = pd.DataFrame()
    batsmen = innings_details.groupby('batsman')
    position = 1
    keys = list(batsmen.groups)
    for batsman in batting_order[batting_order.isin(keys)]:
        batsman_stats = batsmen.get_group(batsman)
        #first_ball = batsman_stats['batsman_runs'][0]
        runs = batsman_stats['batsman_runs'].sum()
        balls = batsman_stats['batsman_runs'].count()-find_extra_balls(batsman_stats)
        dots = find_dots(batsman_stats)
        scoring_shots=find_scoring_shots(batsman_stats)
        byes_legbyes = find_byes_legbyes(batsman_stats)
        boundaries_score = find_boundaries_runs(batsman_stats)
        fours = find_scoring_shots_with(batsman_stats,4)
        sixes = find_scoring_shots_with(batsman_stats,6)
        team = batsman_stats['batting_team'].unique()[0]
        season = matches[matches['id']==batsman_stats['match_id'].unique()[0]]['season']
        batsman_performance = batsman_performance.append(pd.DataFrame({'match':[match],'batsman':[batsman], 'runs': [runs], 'balls':[balls], 'dots':[dots], 'scoring_shots':[scoring_shots], 'boundaries_score': [boundaries_score], 'fours': [fours], 'sixes':[sixes],'innings':inning, 'batting_order':position,'team':[team],'season':season}))
        position = position+1
    return batsman_performance
    
def process_matches(sample,match_performance):
    matches_played = sample.groupby('match_id')
    batsman_performance = pd.DataFrame()
    for match in matches_played.groups:
        innings = matches_played.get_group(match).groupby('inning')
        if 1 in innings.groups:
            batsman_performance=batsman_performance.append(process_inning(match,innings.get_group(1),1, pd.Series(innings.get_group(1)['batsman'].unique())))
        if 2 in innings.groups:
            batsman_performance=batsman_performance.append(process_inning(match,innings.get_group(2),2, pd.Series(innings.get_group(2)['batsman'].unique())))
    batsman_performance.reset_index(drop=True).to_csv('temp.csv')    
    return batsman_performance

match_performance = match_scores(deliveries.head(5000))
player_performance = process_matches(deliveries.head(5000),match_performance)


