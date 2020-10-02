import glob
import pandas
import pandasql

def find_file():
    return glob.glob("../input/**/CrimeMoviesFinal.xlsx", recursive=True)[0]

def run_sql(query):
    return pandasql.sqldf(query, globals())

questions = pandas.read_excel(find_file(), sheet_name="Crime_Movies_2019")
print(questions)
yellow = pandas.read_excel(find_file(), sheet_name="Actors")
print(yellow)
blue = pandas.read_excel(find_file(), sheet_name="Directors")
print(blue)
pink = pandas.read_excel(find_file(), sheet_name="Awards ")
print(pink)


alice = run_sql("""
  select *
    from yellow, questions, pink, blue 
""")


print(alice)

john = run_sql("""
  select Release_Date
    from yellow, questions, pink, blue
""")


print(john)

jackie = run_sql("""
  select Title
    from yellow, questions, pink, blue 
""")


print(jackie)

ken = run_sql("""
  select Staring 
    from yellow, questions, pink, blue
""")


print(ken)