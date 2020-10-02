import glob
import pandas
import pandasql

def find_file():
    return glob.glob("../input/**/*.xlsx", recursive=True)[0]

def run_sql(query):
    return pandasql.sqldf(query, globals())

General = pandas.read_excel(find_file(), sheet_name="General")

print(General)

General = run_sql("""
    select *
    from 'General'
""")

print(General)


Conference = pandas.read_excel(find_file(), sheet_name="Conference")

print(Conference)

Conference = run_sql("""
    select *
    from 'Conference'
""")

print(Conference)


Offensive_Stats = pandas.read_excel(find_file(), sheet_name="Offensive_Stats")

print(Offensive_Stats)

Offensive_Stats = run_sql("""
    select *
    from 'Offensive_Stats'
""")

print(Offensive_Stats)



Defensive_Stats = pandas.read_excel(find_file(), sheet_name="Defensive_Stats")

print(Defensive_Stats)

Defensive_Stats = run_sql("""
    select *
    from 'Defensive_Stats'
""")

print(Defensive_Stats)