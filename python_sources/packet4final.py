import glob
import pandas
import pandasql

def find_file():
    return glob.glob("../input/Packet-4.xlsx", recursive=True)[0]

def run_sql(query):
    return pandasql.sqldf(query, globals())

questions = pandas.read_excel(find_file(), sheet_name="Times")
print(questions)

alice = run_sql("""
    select *
    from Times
    where Times
""")

print(alice)