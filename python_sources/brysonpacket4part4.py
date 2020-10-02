import glob
import pandas
import pandasql

def find_file():
    return glob.glob("../input/**/*.xlsx", recursive=True)[0]

def run_sql(query):
    return pandasql.sqldf(query, globals())

Stats=pandas.read_excel(find_file(), sheet_name="Stats")
print(Stats)

Players=pandas.read_excel(find_file(), sheet_name="Players")
print(Players)

Coaches=pandas.read_excel(find_file(), sheet_name="Coaches")
print(Coaches)

Games=pandas.read_excel(find_file(), sheet_name="Games")
print(Games)

Players=run_sql("""
    select *
    from Stats
    """) 
print(Players)

Players=run_sql("""
    select *
    from Players
    """) 
print(Players)

Players=run_sql("""
    select *
    from Coaches
    """) 
print(Players)

Players=run_sql("""
    select *
    from Games
    """) 
print(Players)