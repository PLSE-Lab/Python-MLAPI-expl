import glob
import pandas
import pandasql

def find_file():
    return glob.glob("../input/**/*.xlsx", recursive=True)[0]

def run_sql(query):
    return pandasql.sqldf(query, globals())

Bachelors = pandas.read_excel(find_file(), sheet_name="Bachelors")
print(Bachelors)

California = run_sql("""
    select *
    from Bachelors
    where ContestantHomeState='California'
""")

print(California)