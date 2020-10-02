#!/usr/bin/env python
# coding: utf-8

# Exploratory data analysis using spark (The code requires spark to run, as there is no server(pyspark) running the code is giving error, hence please ignore the error. I have included the output as markdown. )

# In[ ]:


from pyspark import SparkContext
from pyspark.sql import SQLContext
import pandas as pd
from pandas import DataFrame
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


# In[ ]:


#sc = SparkContext('local','example')  # if using locally
sql_sc = SQLContext(sc)

Spark_Full = sc.emptyRDD()
chunk_100k = pd.read_csv('../input/train.csv', chunksize=100000)
# if you have headers in your csv file:
headers = list(pd.read_csv('../input/test.csv', nrows=0).columns)

for chunky in chunk_100k:  #help was taken from stackeaxhange.com
    Spark_Full +=  sc.parallelize(chunky.values.tolist())

data = Spark_Full.toDF(headers)
data.show()


# +----------+-------------------+--------------------+--------------------+--------------------+------------------+---------------+----+--------------------+-------------------+------------------+
# |Unnamed: 0|        CASE_STATUS|       EMPLOYER_NAME|            SOC_NAME|           JOB_TITLE|FULL_TIME_POSITION|PREVAILING_WAGE|YEAR|            WORKSITE|                lon|               lat|
# +----------+-------------------+--------------------+--------------------+--------------------+------------------+---------------+----+--------------------+-------------------+------------------+
# |         1|CERTIFIED-WITHDRAWN|UNIVERSITY OF MIC...|BIOCHEMISTS AND B...|POSTDOCTORAL RESE...|                 N|        36067.0|2016| ANN ARBOR, MICHIGAN|        -83.7430378|        42.2808256|
# |         2|CERTIFIED-WITHDRAWN|GOODMAN NETWORKS,...|    CHIEF EXECUTIVES|CHIEF OPERATING O...|                 Y|       242674.0|2016|        PLANO, TEXAS|        -96.6988856|        33.0198431|
# |         3|CERTIFIED-WITHDRAWN|PORTS AMERICA GRO...|    CHIEF EXECUTIVES|CHIEF PROCESS OFF...|                 Y|       193066.0|2016|JERSEY CITY, NEW ...|        -74.0776417|        40.7281575|
# |         4|CERTIFIED-WITHDRAWN|GATES CORPORATION...|    CHIEF EXECUTIVES|REGIONAL PRESIDEN...|                 Y|       220314.0|2016|    DENVER, COLORADO|        -104.990251|39.739235799999996|
# |         5|          WITHDRAWN|PEABODY INVESTMEN...|    CHIEF EXECUTIVES|PRESIDENT MONGOLI...|                 Y|       157518.4|2016| ST. LOUIS, MISSOURI|        -90.1994042|        38.6270025|
# |         6|CERTIFIED-WITHDRAWN|BURGER KING CORPO...|    CHIEF EXECUTIVES|EXECUTIVE V P, GL...|                 Y|       225000.0|2016|      MIAMI, FLORIDA|        -80.1917902|        25.7616798|
# |         7|CERTIFIED-WITHDRAWN|BT AND MK ENERGY ...|    CHIEF EXECUTIVES|CHIEF OPERATING O...|                 Y|        91021.0|2016|      HOUSTON, TEXAS| -95.36980279999999|        29.7604267|
# |         8|CERTIFIED-WITHDRAWN|GLOBO MOBILE TECH...|    CHIEF EXECUTIVES|CHIEF OPERATIONS ...|                 Y|       150000.0|2016|SAN JOSE, CALIFORNIA|-121.88632859999998|37.338208200000004|
# |         9|CERTIFIED-WITHDRAWN|  ESI COMPANIES INC.|    CHIEF EXECUTIVES|           PRESIDENT|                 Y|       127546.0|2016|      MEMPHIS, TEXAS|                NaN|               NaN|
# |        10|          WITHDRAWN|LESSARD INTERNATI...|    CHIEF EXECUTIVES|           PRESIDENT|                 Y|       154648.0|2016|    VIENNA, VIRGINIA|        -77.2652604|38.901222499999996|
# |        11|CERTIFIED-WITHDRAWN|  H.J. HEINZ COMPANY|    CHIEF EXECUTIVES|CHIEF INFORMATION...|                 Y|       182978.0|2016|PITTSBURGH, PENNS...|        -79.9958864|40.440624799999995|
# |        12|CERTIFIED-WITHDRAWN|DOW CORNING CORPO...|    CHIEF EXECUTIVES|VICE PRESIDENT AN...|                 Y|       163717.0|2016|   MIDLAND, MICHIGAN| -84.24721159999999|        43.6155825|
# |        13|CERTIFIED-WITHDRAWN|    ACUSHNET COMPANY|    CHIEF EXECUTIVES|   TREASURER AND COO|                 Y|       203860.8|2016|FAIRHAVEN, MASSAC...|                NaN|               NaN|
# |        14|CERTIFIED-WITHDRAWN|       BIOCAIR, INC.|    CHIEF EXECUTIVES|CHIEF COMMERCIAL ...|                 Y|       252637.0|2016|      MIAMI, FLORIDA|        -80.1917902|        25.7616798|
# |        15|CERTIFIED-WITHDRAWN|NEWMONT MINING CO...|    CHIEF EXECUTIVES|        BOARD MEMBER|                 Y|       105914.0|2016|GREENWOOD VILLAGE...|-104.95081409999999|        39.6172101|
# |        16|CERTIFIED-WITHDRAWN|        VRICON, INC.|    CHIEF EXECUTIVES|CHIEF FINANCIAL O...|                 Y|       153046.0|2016|  STERLING, VIRGINIA|        -77.4291298|39.006699299999994|
# |        17|CERTIFIED-WITHDRAWN|CARDIAC SCIENCE C...|  FINANCIAL MANAGERS|VICE PRESIDENT OF...|                 Y|        90834.0|2016| WAUKESHA, WISCONSIN|        -88.2314813|        43.0116784|
# |        18|CERTIFIED-WITHDRAWN|WESTFIELD CORPORA...|    CHIEF EXECUTIVES|GENERAL MANAGER, ...|                 Y|       164050.0|2016|LOS ANGELES, CALI...|-118.24368490000002|        34.0522342|
# |        19|          CERTIFIED|      QUICKLOGIX LLC|    CHIEF EXECUTIVES|                 CEO|                 Y|       187200.0|2016|SANTA CLARA, CALI...|       -121.9552356|37.354107899999995|
# |        20|          CERTIFIED|MCCHRYSTAL GROUP,...|    CHIEF EXECUTIVES|PRESIDENT, NORTHE...|                 Y|       241842.0|2016|ALEXANDRIA, VIRGINIA|        -77.0469214|38.804835499999996|
# +----------+-------------------+--------------------+--------------------+--------------------+------------------+---------------+----+--------------------+-------------------+------------------+
# only showing top 20 rows
# 

# In[ ]:


data.printSchema()


# root
#  |-- Unnamed: 0: long (nullable = true)
#  |-- CASE_STATUS: string (nullable = true)
#  |-- EMPLOYER_NAME: string (nullable = true)
#  |-- SOC_NAME: string (nullable = true)
#  |-- JOB_TITLE: string (nullable = true)
#  |-- FULL_TIME_POSITION: string (nullable = true)
#  |-- PREVAILING_WAGE: double (nullable = true)
#  |-- YEAR: long (nullable = true)
#  |-- WORKSITE: string (nullable = true)
#  |-- lon: double (nullable = true)
#  |-- lat: double (nullable = true)
# 

# In[ ]:


data.head(10) #Previewing the data set


# [Row(Unnamed: 0=1, CASE_STATUS=u'CERTIFIED-WITHDRAWN', EMPLOYER_NAME=u'UNIVERSITY OF MICHIGAN', SOC_NAME=u'BIOCHEMISTS AND BIOPHYSICISTS', JOB_TITLE=u'POSTDOCTORAL RESEARCH FELLOW', FULL_TIME_POSITION=u'N', PREVAILING_WAGE=36067.0, YEAR=2016, WORKSITE=u'ANN ARBOR, MICHIGAN', lon=-83.7430378, lat=42.2808256),
#  Row(Unnamed: 0=2, CASE_STATUS=u'CERTIFIED-WITHDRAWN', EMPLOYER_NAME=u'GOODMAN NETWORKS, INC.', SOC_NAME=u'CHIEF EXECUTIVES', JOB_TITLE=u'CHIEF OPERATING OFFICER', FULL_TIME_POSITION=u'Y', PREVAILING_WAGE=242674.0, YEAR=2016, WORKSITE=u'PLANO, TEXAS', lon=-96.6988856, lat=33.0198431),
#  Row(Unnamed: 0=3, CASE_STATUS=u'CERTIFIED-WITHDRAWN', EMPLOYER_NAME=u'PORTS AMERICA GROUP, INC.', SOC_NAME=u'CHIEF EXECUTIVES', JOB_TITLE=u'CHIEF PROCESS OFFICER', FULL_TIME_POSITION=u'Y', PREVAILING_WAGE=193066.0, YEAR=2016, WORKSITE=u'JERSEY CITY, NEW JERSEY', lon=-74.0776417, lat=40.7281575),
#  Row(Unnamed: 0=4, CASE_STATUS=u'CERTIFIED-WITHDRAWN', EMPLOYER_NAME=u'GATES CORPORATION, A WHOLLY-OWNED SUBSIDIARY OF TOMKINS PLC', SOC_NAME=u'CHIEF EXECUTIVES', JOB_TITLE=u'REGIONAL PRESIDEN, AMERICAS', FULL_TIME_POSITION=u'Y', PREVAILING_WAGE=220314.0, YEAR=2016, WORKSITE=u'DENVER, COLORADO', lon=-104.990251, lat=39.739235799999996),
#  Row(Unnamed: 0=5, CASE_STATUS=u'WITHDRAWN', EMPLOYER_NAME=u'PEABODY INVESTMENTS CORP.', SOC_NAME=u'CHIEF EXECUTIVES', JOB_TITLE=u'PRESIDENT MONGOLIA AND INDIA', FULL_TIME_POSITION=u'Y', PREVAILING_WAGE=157518.4, YEAR=2016, WORKSITE=u'ST. LOUIS, MISSOURI', lon=-90.1994042, lat=38.6270025),
#  Row(Unnamed: 0=6, CASE_STATUS=u'CERTIFIED-WITHDRAWN', EMPLOYER_NAME=u'BURGER KING CORPORATION', SOC_NAME=u'CHIEF EXECUTIVES', JOB_TITLE=u'EXECUTIVE V P, GLOBAL DEVELOPMENT AND PRESIDENT, LATIN AMERI', FULL_TIME_POSITION=u'Y', PREVAILING_WAGE=225000.0, YEAR=2016, WORKSITE=u'MIAMI, FLORIDA', lon=-80.1917902, lat=25.7616798),
#  Row(Unnamed: 0=7, CASE_STATUS=u'CERTIFIED-WITHDRAWN', EMPLOYER_NAME=u'BT AND MK ENERGY AND COMMODITIES', SOC_NAME=u'CHIEF EXECUTIVES', JOB_TITLE=u'CHIEF OPERATING OFFICER', FULL_TIME_POSITION=u'Y', PREVAILING_WAGE=91021.0, YEAR=2016, WORKSITE=u'HOUSTON, TEXAS', lon=-95.36980279999999, lat=29.7604267),
#  Row(Unnamed: 0=8, CASE_STATUS=u'CERTIFIED-WITHDRAWN', EMPLOYER_NAME=u'GLOBO MOBILE TECHNOLOGIES, INC.', SOC_NAME=u'CHIEF EXECUTIVES', JOB_TITLE=u'CHIEF OPERATIONS OFFICER', FULL_TIME_POSITION=u'Y', PREVAILING_WAGE=150000.0, YEAR=2016, WORKSITE=u'SAN JOSE, CALIFORNIA', lon=-121.88632859999998, lat=37.338208200000004),
#  Row(Unnamed: 0=9, CASE_STATUS=u'CERTIFIED-WITHDRAWN', EMPLOYER_NAME=u'ESI COMPANIES INC.', SOC_NAME=u'CHIEF EXECUTIVES', JOB_TITLE=u'PRESIDENT', FULL_TIME_POSITION=u'Y', PREVAILING_WAGE=127546.0, YEAR=2016, WORKSITE=u'MEMPHIS, TEXAS', lon=nan, lat=nan),
#  Row(Unnamed: 0=10, CASE_STATUS=u'WITHDRAWN', EMPLOYER_NAME=u'LESSARD INTERNATIONAL LLC', SOC_NAME=u'CHIEF EXECUTIVES', JOB_TITLE=u'PRESIDENT', FULL_TIME_POSITION=u'Y', PREVAILING_WAGE=154648.0, YEAR=2016, WORKSITE=u'VIENNA, VIRGINIA', lon=-77.2652604, lat=38.901222499999996)]

# In[ ]:


# Impute Missing values
data_new = data.fillna(-1)


# In[ ]:


#Analyzing numerical features
data_new.describe().show()


# +-------+-----------------+-----------------+------------------+------------------+------------------+
# |summary|       Unnamed: 0|  PREVAILING_WAGE|              YEAR|               lon|               lat|
# +-------+-----------------+-----------------+------------------+------------------+------------------+
# |  count|          3002458|          3002458|           3002458|           3002458|           3002458|
# |   mean|        1501229.5|146994.2703409599| 2012.207858028322|-88.87926183308647|36.761799080043886|
# | stddev|866735.1116028281|5287534.656582886|57.650513659531725|25.663494484688528|  8.59502050122896|
# |    min|                1|             -1.0|                -1|      -157.8583333|              -1.0|
# |    max|          3002458|     6.99760672E9|              2016|145.72978909999998|        64.8377778|
# +-------+-----------------+-----------------+------------------+------------------+------------------+
# 
# 

# In[ ]:


#Analyze the H1Bs by the status of their visa applications
data_new.select('CASE_STATUS').show()


# +-------------------+
# |        CASE_STATUS|
# +-------------------+
# |CERTIFIED-WITHDRAWN|
# |CERTIFIED-WITHDRAWN|
# |CERTIFIED-WITHDRAWN|
# |CERTIFIED-WITHDRAWN|
# |          WITHDRAWN|
# |CERTIFIED-WITHDRAWN|
# |CERTIFIED-WITHDRAWN|
# |CERTIFIED-WITHDRAWN|
# |CERTIFIED-WITHDRAWN|
# |          WITHDRAWN|
# |CERTIFIED-WITHDRAWN|
# |CERTIFIED-WITHDRAWN|
# |CERTIFIED-WITHDRAWN|
# |CERTIFIED-WITHDRAWN|
# |CERTIFIED-WITHDRAWN|
# |CERTIFIED-WITHDRAWN|
# |CERTIFIED-WITHDRAWN|
# |CERTIFIED-WITHDRAWN|
# |          CERTIFIED|
# |          CERTIFIED|
# +-------------------+
# only showing top 20 rows
# 

# In[ ]:


data_new.select('CASE_STATUS').distinct().count() #to get distinct case status


# 8

# In[ ]:


data_new.registerTempTable("data_new")
data_new.cache()


# In[ ]:


#Identifying number of visas that are have different case status
data_new.crosstab('EMPLOYER_NAME', 'CASE_STATUS').show()


# +-------------------------+-------------------+---------+---------+------+--------------------------------------------------+-----------+--------+---+
# |EMPLOYER_NAME_CASE_STATUS|CERTIFIED-WITHDRAWN|CERTIFIED|WITHDRAWN|DENIED|PENDING QUALITY AND COMPLIANCE REVIEW - UNASSIGNED|INVALIDATED|REJECTED|NaN|
# +-------------------------+-------------------+---------+---------+------+--------------------------------------------------+-----------+--------+---+
# |     PATHWAY BIOLOGIC,...|                  0|        1|        0|     0|                                                 0|          0|       0|  0|
# |     TCL RESEARCH AMER...|                  0|       14|        1|     0|                                                 0|          0|       0|  0|
# |      UNIVERSITY OF MAINE|                  1|       97|        5|     3|                                                 0|          0|       0|  0|
# |             P2F HOLDINGS|                  0|        2|        0|     0|                                                 0|          0|       0|  0|
# |                M2S, INC.|                  0|       11|        0|     0|                                                 0|          0|       0|  0|
# |        NANOVIRICIDES INC|                  0|        1|        0|     2|                                                 0|          0|       0|  0|
# |     ARIZONA CANCER SP...|                  0|        1|        0|     0|                                                 0|          0|       0|  0|
# |      CLARITY MONEY, INC.|                  0|        1|        0|     0|                                                 0|          0|       0|  0|
# |     W.R. BERKLEY CORP...|                  0|        5|        2|     0|                                                 0|          0|       0|  0|
# |     S. VINODKUMAR USA...|                  0|        4|        0|     0|                                                 0|          0|       0|  0|
# |     CENTRAL FLORIDA B...|                  0|        0|        0|     1|                                                 0|          0|       0|  0|
# |       SPECIAL OPTICS,INC|                  0|        1|        0|     0|                                                 0|          0|       0|  0|
# |        SWEETWATER COUNTY|                  0|        1|        0|     0|                                                 0|          0|       0|  0|
# |       WONDERTREATS, INC.|                  0|        2|        0|     0|                                                 0|          0|       0|  0|
# |     T.K.CHEN INTERNAT...|                  0|        1|        0|     0|                                                 0|          0|       0|  0|
# |     THE GRENADIER COR...|                  0|        1|        0|     0|                                                 0|          0|       0|  0|
# |     MCCONNOR MEADE RI...|                  0|        1|        0|     0|                                                 0|          0|       0|  0|
# |     SONORA COMMUNITY ...|                  0|        1|        0|     0|                                                 0|          0|       0|  0|
# |            JOBHIVE, INC.|                  0|        1|        0|     0|                                                 0|          0|       0|  0|
# |         ARH STUDIOS INC.|                  0|        1|        0|     0|                                                 0|          0|       0|  0|
# +-------------------------+-------------------+---------+---------+------+--------------------------------------------------+-----------+--------+---+
# only showing top 20 rows
# 

# In[ ]:


#Top10 companies getting visa approval (for all the years)
sql_sc.sql("SELECT EMPLOYER_NAME, count(EMPLOYER_NAME) as CERTIFIED_COUNT FROM data_new where CASE_STATUS = 'CERTIFIED' GROUP BY EMPLOYER_NAME order by CERTIFIED_COUNT desc").show(10)


# +--------------------+---------------+
# |       EMPLOYER_NAME|CERTIFIED_COUNT|
# +--------------------+---------------+
# |     INFOSYS LIMITED|         129916|
# |TATA CONSULTANCY ...|          64237|
# |       WIPRO LIMITED|          43476|
# |DELOITTE CONSULTI...|          36120|
# |       ACCENTURE LLP|          32911|
# |IBM INDIA PRIVATE...|          27745|
# |MICROSOFT CORPORA...|          22333|
# |   HCL AMERICA, INC.|          22234|
# |ERNST & YOUNG U.S...|          17874|
# |LARSEN & TOUBRO I...|          16652|
# +--------------------+---------------+
# only showing top 10 rows
# 

# In[ ]:


#Top10 companies getting visa approval (for year 2016)
sql_sc.sql("SELECT EMPLOYER_NAME, count(EMPLOYER_NAME) as CERTIFIED_COUNT FROM data_new where CASE_STATUS = 'CERTIFIED' AND YEAR='2016' GROUP BY EMPLOYER_NAME order by CERTIFIED_COUNT desc").show(10)


# +--------------------+---------------+
# |       EMPLOYER_NAME|CERTIFIED_COUNT|
# +--------------------+---------------+
# |     INFOSYS LIMITED|          25322|
# |CAPGEMINI AMERICA...|          15957|
# |TATA CONSULTANCY ...|          13072|
# |       WIPRO LIMITED|           9528|
# |       ACCENTURE LLP|           9374|
# |IBM INDIA PRIVATE...|           7824|
# |DELOITTE CONSULTI...|           7500|
# |TECH MAHINDRA (AM...|           6681|
# |   HCL AMERICA, INC.|           4917|
# |MICROSOFT CORPORA...|           4669|
# +--------------------+---------------+
# only showing top 10 rows
# 

# In[ ]:


#Worksites for which most number of visas are approved or certified
sql_sc.sql("SELECT WORKSITE, count(*) as Approved FROM data_new where CASE_STATUS = 'CERTIFIED' GROUP BY WORKSITE order by Approved desc").show(5)


# +--------------------+--------+
# |            WORKSITE|Approved|
# +--------------------+--------+
# |  NEW YORK, NEW YORK|  163985|
# |      HOUSTON, TEXAS|   71540|
# |SAN FRANCISCO, CA...|   54336|
# |    ATLANTA, GEORGIA|   46877|
# |   CHICAGO, ILLINOIS|   45249|
# +--------------------+--------+
# only showing top 5 rows
# 

# In[ ]:


#Worksites for which most number of visas are approved or certified in the year 2016
sql_sc.sql("SELECT WORKSITE, count(*) as Approved FROM data_new where CASE_STATUS = 'CERTIFIED' AND YEAR ='2016' GROUP BY WORKSITE order by Approved desc").show(5)


# +--------------------+--------+
# |            WORKSITE|Approved|
# +--------------------+--------+
# |  NEW YORK, NEW YORK|   34639|
# |SAN FRANCISCO, CA...|   13836|
# |      HOUSTON, TEXAS|   13655|
# |    ATLANTA, GEORGIA|   11678|
# |   CHICAGO, ILLINOIS|   11064|
# +--------------------+--------+
# only showing top 5 rows

# In[ ]:


#TOP 5 JOB TITLE for which visa are approved
sql_sc.sql("SELECT JOB_TITLE, count(*) as Approved FROM data_new where CASE_STATUS = 'CERTIFIED' GROUP BY JOB_TITLE order by Approved desc").show(5)


# +-------------------+--------+
# |          JOB_TITLE|Approved|
# +-------------------+--------+
# | PROGRAMMER ANALYST|  222730|
# |  SOFTWARE ENGINEER|  102990|
# |COMPUTER PROGRAMMER|   64018|
# |    SYSTEMS ANALYST|   55744|
# | SOFTWARE DEVELOPER|   37771|
# +-------------------+--------+
# only showing top 5 rows
# 

# In[ ]:


#TOP 5 JOB TITLE for which visa are approved in the year 2016
sql_sc.sql("SELECT JOB_TITLE, count(*) as Approved FROM data_new where CASE_STATUS = 'CERTIFIED' AND YEAR='2016' GROUP BY JOB_TITLE order by Approved desc").show(5)


# +-------------------+--------+
# |          JOB_TITLE|Approved|
# +-------------------+--------+
# | PROGRAMMER ANALYST|   47964|
# |  SOFTWARE ENGINEER|   25890|
# | SOFTWARE DEVELOPER|   12474|
# |    SYSTEMS ANALYST|   10986|
# |COMPUTER PROGRAMMER|   10528|
# +-------------------+--------+
# only showing top 5 rows
# 

# In[ ]:


# As we can see the job title getting most approvals is programmer_analyst
#Lets check which are the company sending most number of programmed analyst and getting approval on H1B Visa 
sql_sc.sql("SELECT EMPLOYER_NAME,count(*) as Approved FROM data_new where CASE_STATUS = 'CERTIFIED' AND JOB_TITLE ='PROGRAMMER ANALYST' GROUP BY EMPLOYER_NAME order by Approved desc").show(5)


# +--------------------+--------+
# |       EMPLOYER_NAME|Approved|
# +--------------------+--------+
# |       WIPRO LIMITED|    7292|
# |SYNTEL CONSULTING...|    2185|
# |RELIABLE SOFTWARE...|    1914|
# |   3I INFOTECH, INC.|    1487|
# |HTC GLOBAL SERVIC...|    1457|
# +--------------------+--------+
# only showing top 5 rows
# 

# In[ ]:


#Lets check which are the company sending most number of programmed analyst and getting approval on H1B Visa in the year 2016
sql_sc.sql("SELECT EMPLOYER_NAME,count(*) as Approved FROM data_new where CASE_STATUS = 'CERTIFIED' AND YEAR='2016' AND JOB_TITLE ='PROGRAMMER ANALYST' GROUP BY EMPLOYER_NAME order by Approved desc").show(5)


# +--------------------+--------+
# |       EMPLOYER_NAME|Approved|
# +--------------------+--------+
# |SYNTEL CONSULTING...|    1409|
# |       WIPRO LIMITED|    1159|
# |HTC GLOBAL SERVIC...|     556|
# |          SYNTEL INC|     524|
# |RELIABLE SOFTWARE...|     463|
# +--------------------+--------+
# only showing top 5 rows
# 
# 

# In[ ]:


#H-1B Salaries Analysis
sql_sc.sql("SELECT EMPLOYER_NAME as businesses, PREVAILING_WAGE as wage, SOC_NAME, JOB_TITLE, YEAR, FULL_TIME_POSITION, CASE_STATUS  FROM data_new where CASE_STATUS ='CERTIFIED' order by PREVAILING_WAGE desc").show(10)


# +--------------------+--------+
# |       EMPLOYER_NAME|Approved|
# +--------------------+--------+
# |SYNTEL CONSULTING...|    1409|
# |       WIPRO LIMITED|    1159|
# |HTC GLOBAL SERVIC...|     556|
# |          SYNTEL INC|     524|
# |RELIABLE SOFTWARE...|     463|
# +--------------------+--------+
# only showing top 5 rows
# 

# In[ ]:


#H-1B Salaries Analysis
sql_sc.sql("SELECT EMPLOYER_NAME as businesses, PREVAILING_WAGE as wage, SOC_NAME, JOB_TITLE, YEAR, FULL_TIME_POSITION, CASE_STATUS  FROM data_new where CASE_STATUS ='CERTIFIED' order by PREVAILING_WAGE desc").show(10)


# +--------------------+------------+--------------------+--------------------+----+------------------+-----------+
# |          businesses|        wage|            SOC_NAME|           JOB_TITLE|YEAR|FULL_TIME_POSITION|CASE_STATUS|
# +--------------------+------------+--------------------+--------------------+----+------------------+-----------+
# |INTEGRATED MEDICA...| 3.0604912E8| Internists, General|RADIATION ONCOLOGIST|2011|                 Y|  CERTIFIED|
# |DEPARTMENT OF VET...| 2.6927472E8|Physicians and Su...|           PHYSICIAN|2011|                 Y|  CERTIFIED|
# | SHELBY HOSPITAL LCC| 2.1839584E8|Family and Genera...| ATTENDING PHYSICIAN|2011|                 Y|  CERTIFIED|
# |         GOOGLE INC.|2.07277824E8|Software Develope...|   SOFTWARE ENGINEER|2012|                 Y|  CERTIFIED|
# |    SOAPROJECTS, INC| 1.8123248E8|ACCOUNTANTS AND A...|SR. MANAGER, SOX ...|2015|                 Y|  CERTIFIED|
# |INFORMATION CONTR...| 1.6950752E8|Computer Systems ...|STAFF CONSULTANT ...|2013|                 Y|  CERTIFIED|
# |              SARVIN| 1.6717168E8|Computer Systems ...| PRINCIPAL ARCHITECT|2011|                 Y|  CERTIFIED|
# |     INFOSYS LIMITED| 1.6163472E8|Computer Systems ...|         MODULE LEAD|2012|                 Y|  CERTIFIED|
# |CONGRUENT SOLUTIO...| 1.5998944E8|Software Develope...|  SOFTWARE DEVELOPER|2012|                 Y|  CERTIFIED|
# |HUMETIS TECHNOLOG...|1.42857728E8|Computer Programmers|  PROGRAMMER ANALYST|2011|                 Y|  CERTIFIED|
# +--------------------+------------+--------------------+--------------------+----+------------------+-----------+
# only showing top 10 rows

# In[ ]:


#Identifying maximim salary by job titles for fulltime position 
sql_sc.sql("SELECT JOB_TITLE ,MAX(PREVAILING_WAGE) as Max_Salary FROM data_new where CASE_STATUS ='CERTIFIED' AND  FULL_TIME_POSITION ='Y' GROUP BY JOB_TITLE ORDER BY Max_Salary DESC").show(10)


# +--------------------+------------+
# |           JOB_TITLE|  Max_Salary|
# +--------------------+------------+
# |RADIATION ONCOLOGIST| 3.0604912E8|
# |           PHYSICIAN| 2.6927472E8|
# | ATTENDING PHYSICIAN| 2.1839584E8|
# |   SOFTWARE ENGINEER|2.07277824E8|
# |SR. MANAGER, SOX ...| 1.8123248E8|
# |STAFF CONSULTANT ...| 1.6950752E8|
# | PRINCIPAL ARCHITECT| 1.6717168E8|
# |         MODULE LEAD| 1.6163472E8|
# |  SOFTWARE DEVELOPER| 1.5998944E8|
# |  PROGRAMMER ANALYST|1.42857728E8|
# +--------------------+------------+
# only showing top 10 rows
# 

# In[ ]:


#Identifying maximim salary by job titles for fulltime position for 2016
sql_sc.sql("SELECT JOB_TITLE ,MAX(PREVAILING_WAGE) as Max_Salary FROM data_new where CASE_STATUS ='CERTIFIED' AND  FULL_TIME_POSITION ='Y' AND YEAR='2016' GROUP BY JOB_TITLE ORDER BY Max_Salary DESC").show(10)


# +--------------------+----------+
# |           JOB_TITLE|Max_Salary|
# +--------------------+----------+
# |    SYSTEMS ANALYSTS|9.119968E7|
# |CHIEF CREATIVE OF...|  631700.0|
# |CARDIOLOGIST PHYS...|  488400.0|
# |INTERVENTIONAL CA...|  413484.0|
# |    ANESTHESIOLOGIST|  395480.0|
# |  EXECUTIVE DIRECTOR|  387132.0|
# |CARDIOLOGIST/INTE...|  350000.0|
# |VISITING ASSISTAN...|  341286.4|
# |        NEUROSURGEON|  338525.0|
# |             SURGEON|  330780.0|
# +--------------------+----------+
# only showing top 10 rows
# 
# 

# In[ ]:


#Identifying maximum salary by employers for fulltime position 
sql_sc.sql("SELECT EMPLOYER_NAME ,MAX(PREVAILING_WAGE) as Max_Salary FROM data_new where CASE_STATUS ='CERTIFIED' AND  FULL_TIME_POSITION ='Y' GROUP BY EMPLOYER_NAME ORDER BY Max_Salary DESC").show(10)


# +--------------------+------------+
# |       EMPLOYER_NAME|  Max_Salary|
# +--------------------+------------+
# |INTEGRATED MEDICA...| 3.0604912E8|
# |DEPARTMENT OF VET...| 2.6927472E8|
# | SHELBY HOSPITAL LCC| 2.1839584E8|
# |         GOOGLE INC.|2.07277824E8|
# |    SOAPROJECTS, INC| 1.8123248E8|
# |INFORMATION CONTR...| 1.6950752E8|
# |              SARVIN| 1.6717168E8|
# |     INFOSYS LIMITED| 1.6163472E8|
# |CONGRUENT SOLUTIO...| 1.5998944E8|
# |HUMETIS TECHNOLOG...|1.42857728E8|
# +--------------------+------------+
# only showing top 10 rows
# 

# In[ ]:


#Identifying maximum salary by employers for fulltime position for 2016
sql_sc.sql("SELECT EMPLOYER_NAME ,MAX(PREVAILING_WAGE) as Max_Salary FROM data_new where CASE_STATUS ='CERTIFIED' AND  FULL_TIME_POSITION ='Y' AND YEAR='2016' GROUP BY EMPLOYER_NAME ORDER BY Max_Salary DESC").show(10)


# +--------------------+----------+
# |       EMPLOYER_NAME|Max_Salary|
# +--------------------+----------+
# |       BIRLASOFT INC|9.119968E7|
# |SAATCHI & SAATCHI...|  631700.0|
# |MERCY PROFESSIONA...|  488400.0|
# |NORTHERN NEVADA M...|  413484.0|
# |EVANGELICAL MEDIC...|  395480.0|
# |VIRTUAL EDUCA FOU...|  387132.0|
# |WESTERN KENTUCKY ...|  350000.0|
# |     PRATT INSTITUTE|  341286.4|
# |MCALLEN HOSPITALI...|  338525.0|
# |         MAYO CLINIC|  330780.0|
# +--------------------+----------+
# only showing top 10 rows
# 
