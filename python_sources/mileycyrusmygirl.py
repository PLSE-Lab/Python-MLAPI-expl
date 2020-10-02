import glob
import pandas
import pandasql
def find_file():
 return glob.glob("../input/**/*.xlsx", recursive=True)[0]
def run_sql(query):
 return pandasql.sqldf(query, globals())
print(find_file())

Celebrities = pandas.read_excel(find_file(),sheet_name="Celebrities")
print(Celebrities)
Songs = pandas.read_excel(find_file(),sheet_name="Songs")
print(Songs)
Movies = pandas.read_excel(find_file(),sheet_name="Movies")
print(Movies)
Shows = pandas.read_excel(find_file(),sheet_name="Shows")
print(Shows)
Albums = pandas.read_excel(find_file(),sheet_name="Albums")
print(Albums)
CelebritiesGender = run_sql("""
 select FirstName,LastName
 from Celebrities
 where GenderID='Female'
""")
print(CelebritiesGender)

MovieID = run_sql("""
 select *
 from Movies
 where Movie_Producer='Clark Spencer'
""")
print(MovieID)

AlbumID = run_sql("""
 select Release_Date
 from Albums
 where AlbumID='Bangerz'
""")
print(AlbumID)