# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import glob
import pandas
import pandasql

def find_file():
    return glob.glob("../input/**/*.xlsx", recursive=True)[0]

def run_sql(query):
    return pandasql.sqldf(query, globals())

print("-------  DATA -------")

player = pandas.read_excel(find_file(), sheet_name="Players")
tournamentplayer = pandas.read_excel(find_file(), sheet_name="Tournamentplayers")
tournament = pandas.read_excel(find_file(), sheet_name="Tournaments")
series = pandas.read_excel(find_file(), sheet_name="Series")

print(player)
print(tournamentplayer)
print(tournament)
print(series)

print("------- query -------")

winnerscore = run_sql("""
    select tournament.TournamentID, min(Playerscore) as winnerscore
    from tournamentplayer, tournament
    where tournamentplayer.TournamentID = tournament.TournamentID
    group by tournament.TournamentID
""")

print(winnerscore)

loser = run_sql("""
    select playername, tournamentplayer.TournamentID
    from winnerscore, tournamentplayer, player
    where winnerscore.TournamentID = tournamentplayer.TournamentID
    and tournamentplayer.playerID = player.playerID
    and Playerscore > winnerscore
""")

print(loser)

lostgame = run_sql("""
    select *
    from loser, tournament, series
    where loser.TournamentID = tournament.TournamentID
    and tournament.SeriesID = series.SeriseID
""")
print(lostgame)
