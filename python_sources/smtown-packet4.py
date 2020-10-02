import glob
import pandas
import pandasql

def find_file():
    return glob.glob("../input/SMTown.xlsx", recursive=True)[0]

def run_sql(query):
    return pandasql.sqldf(query, globals())

IdolGroups = pandas.read_excel(find_file(), sheet_name="IdolGroups")
print(IdolGroups)

BadBoy = run_sql("""
    select *
    from IdolGroups
    where ProgramWinningHitSong='BadBoy'
""")

print(BadBoy)

BestSellingAlbum = pandas.read_excel(find_file(), sheet_name="BestSellingAlbum")
print(BestSellingAlbum)

First = run_sql("""
    select *
    from BestSellingAlbum
    where BillboardWorldAlbumPeakRank='First'
""")

print(First)

HitSong = pandas.read_excel(find_file(), sheet_name="HitSong")
print(HitSong)

One = run_sql("""
    select *
    from HitSong
    where NumberofTheShowWins='One'
""")

print(One)