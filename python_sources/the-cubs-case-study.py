import glob
import pandas
import pandasql

def find_file():
    return glob.glob("../input/**/*.xlsx", recursive=True)[0]

def run_sql(query):
    return pandasql.sqldf(query, globals())

Players = pandas.read_excel(find_file(), sheet_name="Players")
print(Players)

Games = pandas.read_excel(find_file(), sheet_name="Games")
print(Games)

Coaches = pandas.read_excel(find_file(), sheet_name="Coaches")
print(Coaches)

Stats = pandas.read_excel(find_file(), sheet_name="Stats")
print(Stats)


Players = run_sql("""
    select Players.PlayerName, Stats.GamesStarted
    from Players
    INNER JOIN Stats ON Players.PlayerID=Stats.PlayerID
    """)

print(Players)

Games = run_sql("""
    select *
    from Games
    """)
print(Games)

Coaches = run_sql("""
    select Coaches.CoachName, Coaches.Role
    from Coaches
    """)
print(Coaches)

Stats = run_sql("""
    select *
    from Stats
    """)
print(Stats)



