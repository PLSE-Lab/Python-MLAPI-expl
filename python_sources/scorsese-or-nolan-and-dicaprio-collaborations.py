import csv as csv

readdata = csv.DictReader(open("../input/movie_metadata.csv"))

for row in readdata:
    if "Scorsese" in row["director_name"] or "Nolan" in row["director_name"]:
        if "DiCaprio" in row["actor_1_name"]:
            print (row["movie_title"])

