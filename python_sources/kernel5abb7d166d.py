import glob
import pandas
import pandasql

def find_file():
    return glob.glob("../input/**/*.xlsx", recursive=True)[0]

def run_sql(query):
    return pandasql.sqldf(query, globals())

general_background = pandas.read_excel(find_file(), sheet_name="GeneralBackground")
print(general_background)

origin = pandas.read_excel(find_file(), sheet_name="Origin")
print(origin)

finished = pandas.read_excel(find_file(), sheet_name="Finished")
print(finished)

general_background = run_sql("""
    select *
    from general_background
""")

print(origin)