import glob
import pandas
import pandasql
def find_file():
 return glob.glob("../input/**/*.xlsx", recursive=True)[0]
def run_sql(query):
 return pandasql.sqldf(query, globals())
artists = pandas.read_excel(find_file(), sheet_name="Artist")
print(artists)
ed_sheeran = run_sql("""
 select *
 from artists
 where Artist_Name='Ed Sheeran'
""")
print(ed_sheeran)

youngest_artist = run_sql("""
 select min(Artist_Age) as youngest_artist_age
 from artists
""")
print(youngest_artist)

more_than_50_million_montly_listeners = run_sql("""
 select *
 from artists
 where Listeners_Per_Month > 50
""")
print(more_than_50_million_montly_listeners)

least_listeners = run_sql("""
 select min(Listeners_Per_Month) as least_listeners
 from artists
""")
print(least_listeners)