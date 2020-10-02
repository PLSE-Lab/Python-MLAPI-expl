import glob
import pandas
import pandasql

def find_file():
    return glob.glob("../input/Packet4.xlsx", recursive=True)[0]

def run_sql(query):
    return pandasql.sqldf(query, globals())
Film = pandas.read_excel(find_file(), sheet_name="film")
Series = pandas.read_excel(find_file(), sheet_name="series")
Director = pandas.read_excel(find_file(), sheet_name="dirctor")
Character = pandas.read_excel(find_file(), sheet_name="character")
Casting= pandas.read_excel(find_file(), sheet_name="casting")

print(Film)
print(Series)
print(Casting)
print(Character)
print(Director)

Thor = run_sql("""
    select *
    from Series
    where FilmSeries='Thor'
""")

WhichFilmIsThor = run_sql("""
    select FilmName, FilmSeries
    from film, series
    where Film.SeriesID=Series.SeriesID
""")

TaikaWaititi = run_sql("""
    select *
    from Director
    where DirectorName='TaikaWaititi'
""")

WhoIsBlackWidow = run_sql("""
    select CharacterCasting, CastingName
    from character, casting
    where Character.CharacterCasting=Casting.CharacterCasting
""")

print(Thor)
print(WhichFilmIsThor)
print(TaikaWaititi)
print(WhoIsBlackWidow)