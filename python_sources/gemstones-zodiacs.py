import glob
import pandas
import pandasql

def find_file():
    return glob.glob("../input/**/*.xlsx", recursive=True)[0]

def run_sql(query):
    return pandasql.sqldf(query, globals())

"----JANUARY TABLE----"

Zodiacs = pandas.read_excel(find_file(), sheet_name="Zodiacs")
print(Zodiacs)

January = pandas.read_excel(find_file(), sheet_name="January")
print(January)

Zodiacs = run_sql("""
    select *
    from Zodiacs, January
    where Zodiacs.MonthID='January.MonthID'
""")

"----FEBRUARY TABLE----"

print(Zodiacs)

Zodiacs = pandas.read_excel(find_file(), sheet_name="Zodiacs")
print(Zodiacs)

February = pandas.read_excel(find_file(), sheet_name="February")
print(February)

Zodiacs = run_sql("""
    select *
    from Zodiacs, February
    where Zodiacs.MonthID='February.MonthID'
""")

"----MARCH TABLE----"

print(Zodiacs)

Zodiacs = pandas.read_excel(find_file(), sheet_name="Zodiacs")
print(Zodiacs)

March = pandas.read_excel(find_file(), sheet_name="March")
print(March)

Zodiacs = run_sql("""
    select *
    from Zodiacs, March
    where Zodiacs.MonthID='March.MonthID'
""")

"----APRIL TABLE----"

print(Zodiacs)

Zodiacs = pandas.read_excel(find_file(), sheet_name="Zodiacs")
print(Zodiacs)

April = pandas.read_excel(find_file(), sheet_name="April")
print(April)

Zodiacs = run_sql("""
    select *
    from Zodiacs, April
    where Zodiacs.MonthID='April.MonthID'
""")

"----MAY TABLE----"

print(Zodiacs)

Zodiacs = pandas.read_excel(find_file(), sheet_name="Zodiacs")
print(Zodiacs)

May = pandas.read_excel(find_file(), sheet_name="May")
print(May)

Zodiacs = run_sql("""
    select *
    from Zodiacs, May
    where Zodiacs.MonthID='May.MonthID'
""")

"----JUNE TABLE----"

print(Zodiacs)

Zodiacs = pandas.read_excel(find_file(), sheet_name="Zodiacs")
print(Zodiacs)

June = pandas.read_excel(find_file(), sheet_name="June")
print(June)

Zodiacs = run_sql("""
    select *
    from Zodiacs, June
    where Zodiacs.MonthID='June.MonthID'
""")

"----JULY TABLE----"

print(Zodiacs)

Zodiacs = pandas.read_excel(find_file(), sheet_name="Zodiacs")
print(Zodiacs)

July = pandas.read_excel(find_file(), sheet_name="July")
print(July)

Zodiacs = run_sql("""
    select *
    from Zodiacs, July
    where Zodiacs.MonthID='July.MonthID'
""")

"----AUGUST TABLE----"

print(Zodiacs)

Zodiacs = pandas.read_excel(find_file(), sheet_name="Zodiacs")
print(Zodiacs)

August = pandas.read_excel(find_file(), sheet_name="August")
print(August)

Zodiacs = run_sql("""
    select *
    from Zodiacs, August
    where Zodiacs.MonthID='August.MonthID'
""")

"----SEPTEMBER TABLE----"

print(Zodiacs)

Zodiacs = pandas.read_excel(find_file(), sheet_name="Zodiacs")
print(Zodiacs)

September = pandas.read_excel(find_file(), sheet_name="September")
print(September)

Zodiacs = run_sql("""
    select *
    from Zodiacs, September
    where Zodiacs.MonthID='September.MonthID'
""")

"----OCTOBER TABLE----"

print(Zodiacs)

Zodiacs = pandas.read_excel(find_file(), sheet_name="Zodiacs")
print(Zodiacs)

October = pandas.read_excel(find_file(), sheet_name="October")
print(October)

Zodiacs = run_sql("""
    select *
    from Zodiacs, October
    where Zodiacs.MonthID='October.MonthID'
""")

"----NOVEMBER TABLE----"

print(Zodiacs)

Zodiacs = pandas.read_excel(find_file(), sheet_name="Zodiacs")
print(Zodiacs)

November = pandas.read_excel(find_file(), sheet_name="November")
print(November)

Zodiacs = run_sql("""
    select *
    from Zodiacs, November
    where Zodiacs.MonthID='November.MonthID'
""")

"----DECEMBER TABLE----"

print(Zodiacs)

Zodiacs = pandas.read_excel(find_file(), sheet_name="Zodiacs")
print(Zodiacs)

December = pandas.read_excel(find_file(), sheet_name="December")
print(December)

Zodiacs = run_sql("""
    select *
    from Zodiacs, December
    where Zodiacs.MonthID='December.MonthID'
""")

