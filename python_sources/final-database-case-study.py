import glob
import pandas
import pandasql

def find_file():
    return glob.glob("../input/finalcasestudy.xlsx", recursive=True)[0]

def run_sql(query):
    return pandasql.sqldf(query, globals())

players = pandas.read_excel(find_file(), sheet_name="Players")
print(players)

teams = pandas.read_excel(find_file(), sheet_name="Teams")
print(teams)

coaches = pandas.read_excel(find_file(), sheet_name="Coaches")
print(coaches)

positions = pandas.read_excel(find_file(), sheet_name="Positions")
print(positions)

goals = pandas.read_excel(find_file(), sheet_name="Goals")
print(goals)

player_goals = run_sql("""
    select PlayerName, GoalNumber
    from players, goals
    where players.PlayerID = goals.PlayerID
""")

player_positions = run_sql("""
    select PlayerName, PositionName
    from players, positions
    where players.PositionID = positions.PositionID
""")

player_coaches = run_sql("""
    select PlayerName, CoachName
    from players, coaches
    where players.CoachID = coaches.CoachID
""")

player_team = run_sql("""
    select PlayerName, TeamName
    from players, teams
    where players.TeamID = teams.TeamID
""")

print(player_goals)
print(player_positions)
print(player_coaches)
print(player_team)