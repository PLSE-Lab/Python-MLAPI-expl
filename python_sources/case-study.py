import glob
import pandas
import pandasql
def find_file():
 return glob.glob("../input/**/*.xlsx", recursive=True)[0]
def run_sql(query):
 return pandasql.sqldf(query, globals())
print(find_file())

Shows = pandas.read_excel(find_file(),sheet_name="Shows")
print(Shows)
Channels = pandas.read_excel(find_file(),sheet_name="Channels")
print(Channels)
Viewers = pandas.read_excel(find_file(),sheet_name="Viewers")
print(Viewers)
Actors = pandas.read_excel(find_file(),sheet_name="Actors")
print(Actors)
ShowsProducer = run_sql("""
 select *
 from Shows
 where Show_Producer='Ryan Seacrest'
""")
print(ShowsProducer)

ActorsAge = run_sql("""
 select FirstName,LastName
 from Actors
 where Actor_Age='25'
""")
print(ActorsAge)

ViewersGender = run_sql("""
 select ViewerNameID
 from Viewers
 where GenderID='Male'
""")
print(ViewersGender)
 