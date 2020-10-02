import glob
import pandas
import pandasql

def find_file():
    return glob.glob("../input/Book22.xlsx", recursive=True)[0]

def run_sql(query):
    return pandasql.sqldf(query, globals())

BattingStats = pandas.read_excel(find_file(), sheet_name="BattingStats")
print(BattingStats)
Players = pandas.read_excel(find_file(), sheet_name="Players")
print(Players)
Positions = pandas.read_excel(find_file(), sheet_name="Positions")
print(Positions)
Games = pandas.read_excel(find_file(), sheet_name="Games")
print(Games)
Homeruns = pandas.read_excel(find_file(), sheet_name="Homeruns")
print(Homeruns)

stats30 = run_sql("""
    select *
    from BattingStats
    where NumberOfAtBats = "30"
""")
print(stats30)

runs = run_sql("""
    select *
    from BattingStats
    where RunsScored = "0"
""")
print(runs)

PlayerHR = run_sql("""
    select PlayerName
    from Players, Homeruns
    where Players.PlayerID = Homeruns.PlayerID
""")
print(PlayerHR)

games = run_sql("""
    select PlayerID, GamesPlayed
    from BattingStats
    where GamesPlayed = "7"
""")
print(games)

