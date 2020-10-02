import glob

import pandas

import pandasql

def find_file():

    return glob.glob("../input/**/*.xlsx", recursive=True)[0]

def run_sql(query):

    return pandasql.sqldf(query, globals())


Physical = pandas.read_excel(find_file(), sheet_name="Physical")

print(Physical)

N = run_sql("""

    select *

    from Physical

    where Living='N'

""")
print(N)



Demographics = pandas.read_excel(find_file(), sheet_name="Demographics")

print(Demographics)

England = run_sql("""

    select count(*)

    from Demographics

    where WhereFromCountry='England'

""")
print(England)



CouplingInfo = pandas.read_excel(find_file(), sheet_name="CouplingInfo")

print(CouplingInfo)

N = run_sql("""

    select count(*)

    from CouplingInfo

    where WithEndPartner='N'

""")
print(N)



Physical = pandas.read_excel(find_file(), sheet_name="Physical")

print(Physical)

w = run_sql("""

    select count(*)

    from Physical

    where Ethnicity='w'

""")
print(w)