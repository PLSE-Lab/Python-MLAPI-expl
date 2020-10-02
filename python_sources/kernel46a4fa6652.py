import glob
import pandas
import pandasql

def find_file():
    return glob.glob("../input/**/*.xlsx", recursive=True)[0]

def run_sql(query):
    return pandasql.sqldf(query, globals())

DogBreeds = pandas.read_excel(find_file(), sheet_name="DogBreeds")
print(DogBreeds)

DogID = run_sql("""
    select *
    from DogBreeds
""")

print(DogID)

Maltese = pandas.read_excel(find_file(), sheet_name="Maltese")
print(Maltese)

DogID = run_sql("""
    select *
    from Maltese
""")

print(DogID)

run_sql("""
    select DogName
    from Maltese
    where DogFurColor = "White"
""")