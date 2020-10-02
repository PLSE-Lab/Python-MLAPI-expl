import glob
import Bachelors
import Bachelorssql

def find_file():
    return glob.glob("../input/**/*.xlsx", recursive=True)[0]

def run_sql(query):
    return Bachelorssql.sqldf(query, globals())

Bachelors = pandas.read_excel(find_file(), sheet_name="Bachelors")
print(Bachelors)

TylerC = run_sql("""
    select *
    from Success
    where SuccessInsta='>50,000'
""")

print(TylerC)