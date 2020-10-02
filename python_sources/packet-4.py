import glob
import pandas
import pandasql

def find_file():
     return glob.glob("../input/**/*.xlsx", recursive=True)[0]

def run_sql(query):
     return pandasql.sqldf(query, globals())

Skis = pandas.read_excel(find_file(), sheet_name="Skis")
print(Skis)

Boots = pandas.read_excel(find_file(), sheet_name="Boots")
print(Boots)

Poles = pandas.read_excel(find_file(), sheet_name="Poles")
print(Poles)
   
Bridge = pandas.read_excel(find_file(), sheet_name="Bridge")
print(Bridge)
       
Mountain = pandas.read_excel(find_file(), sheet_name="Mountain")
print(Mountain)

MtnInfo = pandas.read_excel(find_file(), sheet_name="MtnInfo")
print(MtnInfo)

Trails = pandas.read_excel(find_file(), sheet_name="Trails")
print(Trails)

Advanced = run_sql("""
    select *
    from Boots
    where Boots.Difficulty='Advanced'
""")

print(Advanced)

SkiBrand1 = run_sql("""
    select *
    from Skis, Boots
    where Skis.Brand_ID=Boots.Brand_ID and Skis.Brand_ID=1
""")

print(SkiBrand1)

Mountain102 = run_sql("""
    select *
    from MtnInfo, Trails
    where MtnInfo.MountainID=Trails.MountainID and MtnInfo.MountainID=102
""")



print(Mountain102)




