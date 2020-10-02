#!/usr/bin/env python
# coding: utf-8

# **I TRIED TO USE ALL USEFUL FUNCTION TO UNDERSTAND PANDAS**

# In[ ]:


import pandas as pd

#imported pandas module

"""
i downloaded datas from kaggle as csv file and i put that file in my pycharm work project folder to read it here!

i will also upload csv file here



"""

# HOW TO READ DATA

# lets read out dataset first( you can see here " how to read datasets"

soccer = pd.read_csv("mlsSalaries.csv")

print(soccer)

"""
    club last_name    first_name position  base_salary  guaranteed_compensation
0    ATL   Almiron        Miguel        M    1912500.0               2297000.00
1    ATL   Ambrose         Mikey        D      65625.0                 65625.00
2    ATL      Asad         Yamil        M     150000.0                150000.00
3    ATL     Bloom          Mark        D      99225.0                106573.89
4    ATL  Carleton        Andrew        F      65000.0                 77400.00
..   ...       ...           ...      ...          ...                      ...
611  VAN  Tornaghi         Paolo       GK      80000.0                 80000.00
612  VAN    Waston       Kendall        D     350000.0                368125.00
613  VAN  Williams       Sheanon        D     175000.0                184000.00
614  NaN   Babouli            Mo        F      54075.0                 54075.00
615  NaN    Ramajo  David Mateos        D     420000.0                453333.33

[616 rows x 6 columns]

you can see all uotputs right under the code like this

the reason im doing this make works look better and understandable 
also you can print out this works and you can work on paper

"""
# HOW TO READ FIRST 10 DATA


# if you wanna read first 10 data: " head() " function
# if you dont give "n=" parameter in the head function its gonna take automatically first 5 data

print(soccer.head(n=10))


""" club       last_name  ... base_salary guaranteed_compensation
0  ATL         Almiron  ...   1912500.0              2297000.00
1  ATL         Ambrose  ...     65625.0                65625.00
2  ATL            Asad  ...    150000.0               150000.00
3  ATL           Bloom  ...     99225.0               106573.89
4  ATL        Carleton  ...     65000.0                77400.00
5  ATL         Carmona  ...    675000.0               725000.00
6  ATL           Garza  ...    150000.0               150000.00
7  ATL  Gonzalez Pirez  ...    250008.0               285008.00
8  ATL          Goslin  ...     70000.0                74000.00
9  ATL         Gressel  ...     75000.0                93750.00

[10 rows x 6 columns]

first 10 datas 
"""

# FINDING HOW MANY DATAS IN CSV FILE(counting index(row))

print(len(soccer.index))

"""
616 

ther are 616 rows in this csv file

"""

# FINDING AVARAGE SALARY FROM ALL TOTAL

# first thing we need to take base_salary column as series and then we will use "mean()" function

print(soccer["base_salary"].mean())

"""
296977.7368668831

this is the avarage of salaries

"""

# FINDING HIGEST SALARY

# we need to take "base_salary" column again to find highest salary

# we will use "max()" function

print(soccer["base_salary"].max())

"""
6660000.0

this is the highest salary   

"""

# HOW TO FIND THE SURNAME WHO HAS HIGHEST GUARANTEED COMPENSATION(TAZMINAT

#first step we need to take "guaranteed_compensation" column and then we will make filter on this

print(soccer[soccer["guaranteed_compensation"] == soccer["guaranteed_compensation"].max()]["last_name"])

"""
401    Kaka
Name: last_name, dtype: object

we found as Kaka who has the highest "guaranteed_compensation"

"""

# if you guys wanna make this look better you can take only name as " kaka " with "iloc[]" function

# you can take any index with iloc[] function


print(soccer[soccer["guaranteed_compensation"] == soccer["guaranteed_compensation"].max()]["last_name"].iloc[0])

"""
Kaka

the only output will be just  " kaka "

we took index "0" here

"""


# HOW TO FIND GONZALES PIREZ's POSITION

# firs step here we need to do: filter as "last_name"


print(soccer[soccer["last_name"] == "Gonzalez Pirez"])

"""

club       last_name  first_name position   base_salary guaranteed_compensation
7  ATL  Gonzalez Pirez  Leonardo    D     250008.0                285008.0


we can see information about Gonzalez Pirez here

"""
# second step here: taking "position" column and then taking index as "0" which is "D"


print(soccer[soccer["last_name"] == "Gonzalez Pirez"]["position"].iloc[0])


"""
D

output will be "D" (defanse) here 

"""

# HOW TO MAKE SOCCER PLAYERS GROUP BY POSITION AND FIND AVARAGE SALARIES WITH THIS POSITIONS]
#we will use "groupby" function here

print(soccer.groupby("position").mean())

"""
          base_salary  guaranteed_compensation
position                                        
D         166574.093784            179533.184811
D-M       134930.000000            154328.927857
F         489588.598349            557437.206514
F-M       335834.000000            367920.805000
F/M       125000.000000            131250.000000
GK        146472.515538            158665.155692
M         376106.218632            406781.788396
M-D       219032.250000            230282.250000
M-F       212975.604211            231104.551579
M/F       550000.000000            565000.000000

"""

# HOW TO FIND HOW MANY POSITION IN DATASET

# we can use "nunique()" function here

print(soccer["position"].nunique())

"""

10
output will be "10"

there are 10 different(unique) position in dataset

"""

# HOW TO FIND: HOW MANY SOCCER PLAYER IN EVERY SINGLE POSITION

# we can use "value_counts()" function here:

print(soccer["position"].value_counts())

"""
M      212
D      185
F      109
GK      65
M-F     19
D-M     14
F-M      6
M-D      4
F/M      1
M/F      1
Name: position, dtype: int64



"""


# HOW TO FIND: HOW MANY SOCCER PLAYER PLAYING IN EVERY SINGLE TIME

# we will do "value_counts()" here as taking "club" column

print(soccer["club"].value_counts())

"""
VAN      32
PHI      31
ATL      31
ORL      30
CLB      30
DAL      29
SJ       29
NYCFC    28
HOU      28
NYRB     28
CHI      27
POR      27
MTL      27
MNUFC    27
DC       27
TOR      27
LA       27
RSL      27
COL      26
KC       26
SEA      25
NE       23
LAFC      2
Name: club, dtype: int64

"""


# HOW TO FIND: PLAYERS WHO HAS "SON" INSIDE OF THEIR SURNAMES

#first step we need write a function here:

def son_find(last_name):
    if "son" in last_name.lower():
        return True
    return False

# we can use "apply()" function to use that we created function "son_find"

print(soccer[soccer["last_name"].apply(son_find)])


"""
      club      last_name  ... base_salary guaranteed_compensation
21     ATL       Peterson  ...   165300.00               165300.00
23     ATL       Robinson  ...   125000.04               195000.04
45     CHI        Johnson  ...    65004.00                65004.00
48     CHI        Lampson  ...    76000.00                81375.00
85     CLB        Swanson  ...    85000.00               105416.67
163     DC       Robinson  ...    69996.00                69996.00
233     LA       Jamieson  ...    66150.00                66150.00
265  MNUFC          Ibson  ...   200004.00               210337.40
276  MNUFC       Thiesson  ...   171000.00               210166.67
295    MTL  Jackson-Hamel  ...    66150.00                66150.00
327     NE         Watson  ...   150000.00               155666.67
338  NYCFC       Harrison  ...   130000.00               165500.00
341  NYCFC        Johnson  ...   220008.00               220008.00
400    ORL        Johnson  ...   414000.00               450000.00
442    PHI        Simpson  ...   465000.00               508333.33
459    POR        Gleeson  ...   110000.00               115166.67
522    SEA       Svensson  ...   170000.04               170000.04
550     SJ       Thompson  ...   150000.00               155000.00
594    VAN       Jacobson  ...   150000.00               175000.00

[19 rows x 6 columns]


"""

















# In[ ]:


import pandas as pd
mlsSalaries copy = pd.read_csv("../input/mlsSalaries copy.csv")

