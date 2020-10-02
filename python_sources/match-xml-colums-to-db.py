import sqlite3
import datetime
import csv
import xml.etree.ElementTree as ET

def get_xml_attribute(xml,element):

    try:

        return xml.find(element).text

    except AttributeError:

        return 'NA'

def get_fields(numoffields):

    col = ''.join("?," for i in range(numoffields))
    return col[0:len(col)-1]

fieldconfig = (ET.parse(r'C:/Users/shyam.ashar/Desktop/soccer/fieldconfig_possession.xml')).getroot()
source_fields = (fieldconfig.find("sourcefields").text).split(sep=",")

source_key = fieldconfig.find("sourcekeycolumn").text
source_xml = fieldconfig.find("sourcexmlcolumn").text

target_table = fieldconfig.find("targettable").text

#print(datetime.datetime.now().time())
conn = sqlite3.connect(r'../input/database.sqlite')
c = conn.cursor()

#c.execute("delete from Goals")

query_string = "select " + source_key + "," + source_xml + " from match where season > '2011/2012'"

c.execute(query_string)

#c.execute("SELECT id,goal from Match where season > '2011/2012'")
#source_fields = ['comment','stats','event_incident_typefk','elapsed','elapsed_plus','player2','subtype','player1','sortorder','team','id','n','type','goal_type']

allgoalincidents = []

for item in c.fetchall():

    try:

        goals = (ET.ElementTree(ET.fromstring(item[1]))).getroot()

        for goal in goals:

            matchgoalincidents = []
            matchgoalincidents.append(item[0])
            
            for incidentattribute in source_fields:
                matchgoalincidents.append(get_xml_attribute(goal,incidentattribute))

            allgoalincidents.append(matchgoalincidents)
            
    except:
        
        pass

print(len(allgoalincidents))

insertcolums = get_fields(len(fieldconfig.find("targetfields").text.split(sep=",")))

query_string = "Insert into " + target_table + " values (" + insertcolums + ")"
c.executemany(query_string,allgoalincidents)

#c.executemany("Insert into Goals values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",allgoalincidents)
#conn.commit()

    
    





