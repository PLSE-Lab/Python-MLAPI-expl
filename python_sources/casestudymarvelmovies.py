import glob
import pandas
import pandasql

def find_file():
    return glob.glob("../input/**/*.xlsx", recursive=True)[0]

def run_sql(query):
    return pandasql.sqldf(query, globals())

Actors = pandas.read_excel(find_file(), sheet_name="Actors")
MoviesInfo = pandas.read_excel(find_file(), sheet_name="MoviesInfo")
Directors = pandas.read_excel(find_file(), sheet_name="Directors")
print(MoviesInfo)
print("------- MoviesInfo DATA -------")
print(Actors)
print("------- Actors DATA -------")
print(Directors)
print("------- Directors DATA -------")

actorsInMovies = run_sql("""
    select Name
    from Actors
    where Actors.MovieID = 605 or Actors.MovieID1 = 605 or Actors.MovieID2 = 605 or Actors.MovieID3 = 605
""")

money = run_sql("""
    select WorldWideTotal, Title
    from MoviesInfo
    where WorldWideTotal > 1500000000
""")

setting = run_sql("""
    select MovieSettingYear, Title
    from MoviesInfo
    where MovieSettingYear >= 2015
""")

age = run_sql("""
    select Names, Age
    from Directors
    where Age < 40
""")

directors = run_sql("""
    select AmountOfMovies, Names
    from Directors
    where AmountOfMovies > 1
""")


print(actorsInMovies)
print(money)
print(setting)
print(age)
print(directors)