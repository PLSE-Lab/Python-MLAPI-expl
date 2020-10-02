import glob
import pandas
import pandasql

def find_file():
    return glob.glob("../input/lis project.xlsx", recursive=True)[0]

def run_sql(query):
    return pandasql.sqldf(query, globals())

teams = pandas.read_excel(find_file(), sheet_name="Teams")
print(teams)

MVPs = pandas.read_excel(find_file(), sheet_name="MVPs")
print(MVPs)

PlayoffAppearances = pandas.read_excel(find_file(), sheet_name="PlayoffAppearances")
print(PlayoffAppearances)

Championships = pandas.read_excel(find_file(), sheet_name="Championships")
print(Championships)

LakerMVP = run_sql("""
    select *
    from MVPs
    where TeamID='2'
""")

print(LakerMVP)

WarriorMVP = run_sql("""
    select *
    from MVPs
    where TeamID='4'
""")

print(WarriorMVP)

ClipperMVP = run_sql("""
    select *
    from MVPs
    where TeamID='1'
""")

print(ClipperMVP)

KingMVP = run_sql("""
    select *
    from MVPs
    where TeamID='3'
""")

print(KingMVP)





