import glob
import pandas
import pandasql

def find_file():
    return glob.glob("../input/**/*.xlsx", recursive=True)[0]

def run_sql(query):
    return pandasql.sqldf(query, globals())

print(find_file())

Songs = pandas.read_excel(find_file(), sheet_name="Songs")
print(Songs)
Albums = pandas.read_excel(find_file(), sheet_name="Albums")
print(Albums)
Performances = pandas.read_excel(find_file(), sheet_name="Performances")
print(Performances)
Artists = pandas.read_excel(find_file(), sheet_name="Artists")
print(Artists)
PerformanceSongs = pandas.read_excel(find_file(), sheet_name="PerformanceSongs")
print(PerformanceSongs)
Collaborators = pandas.read_excel(find_file(), sheet_name="Collaborators")
print(Collaborators)

SongsinAlbums = run_sql("""
    select SongName, AlbumName
    from Songs, Albums
    where Albums.AlbumID = Songs.AlbumID
""")

print(SongsinAlbums)

FemaleArtists = run_sql("""
    select ArtistID, ArtistGender
    from Artists
    where ArtistGender = "F"
""")

print(FemaleArtists)

HipHopArtists = run_sql("""
    select *
    From Artists
    where ArtistGenre = "Hip Hop"
""")

print(HipHopArtists)

HipHopArtists = run_sql("""
    select *
    From Songs
    where AlbumID = "2"
""")

print(HipHopArtists)
