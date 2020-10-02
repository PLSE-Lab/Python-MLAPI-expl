import glob
import pandas
import pandasql

def find_file():
    return glob.glob("../input/**/Packet4Data.xlsx", recursive=True)[0]

def run_sql(query):
    return pandasql.sqldf(query, globals())

Players = pandas.read_excel(find_file(), sheet_name="Players")
print(Players)
Teams = pandas.read_excel(find_file(), sheet_name="Teams")
print(Teams)
Sponsors = pandas.read_excel(find_file(), sheet_name="Sponsors")
print(Sponsors)
Positions = pandas.read_excel(find_file(), sheet_name="Positions")
print(Positions)
Scores = pandas.read_excel(find_file(), sheet_name="Scores")
print(Scores)

alice = run_sql("""
    select *
    from questions
    where Question_Asked_By='Alice'
""")
