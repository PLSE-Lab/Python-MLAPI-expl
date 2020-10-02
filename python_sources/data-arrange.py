# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Any results you write to the current directory are saved as output.
# Imports
import sqlite3 as lite
import sys
import matplotlib.pyplot as plt
import pylab
plt.rcdefaults()

# Constants
DATABASE = '../input/database.sqlite'

STATE_COUNT_QUERY = 'SELECT StateCode, COUNT(DISTINCT StandardComponentId) AS Count FROM BenefitsCostSharing GROUP BY' \
                    ' StateCode ORDER BY COUNT(DISTINCT StandardComponentId) DESC'

AGE_RATE_QUERY = 'SELECT StateCode, Age, CASE WHEN Age IS "Family Option" THEN ' \
                 'AVG((IndividualRate+Couple+PrimarySubscriberAndOneDependent+PrimarySubscriberAndTwoDependents+' \
                 'PrimarySubscriberAndThreeOrMoreDependents+CoupleAndOneDependent+CoupleAndTwoDependents+' \
                 'CoupleAndThreeOrMoreDependents)/8) ELSE AVG (IndividualRate) END AS AvgIndividualRate FROM ' \
                 '(SELECT * FROM Rate WHERE IndividualRate < 9999 AND IndividualRate > 0) Rate2 GROUP BY StateCode, ' \
                 'Age ORDER BY StateCode, Age DESC'


# Functions
def query_for_list(con, query):
    con.row_factory = lite.Row

    cur = con.cursor()
    execute = cur.execute(query)

    rows = cur.fetchall()
    return {"desc": execute.description, "rows": rows}


def state_benefits_cost_sharing(con):
    result = query_for_list(con, STATE_COUNT_QUERY)
    rows = result['rows']
    width = 0.75
    tick_positions = list(range(len(rows)))
    states = [str(row[0]) for row in rows]
    counts = [int(row[1]) for row in rows]

    f = open('state_benefits_cost_sharing.txt', 'w')

    f.write('State\t\tCount\n')
    for s, c in zip(states, counts):
        f.write(s + '\t\t' + str(c)+'\n')
    f.close()

    plt.clf()
    font = {'size': 8}
    plt.rc('font', **font)
    plt.bar(tick_positions, counts, bottom=0.07, width=width)

    plt.xticks(tick_positions, states)
    plt.xlabel('State')
    plt.xticks(rotation=90)
    plt.ylabel('Number of benefits')
    plt.title('Benefits Cost Sharing/State')
    pylab.savefig('STATE_COUNT.png', bbox_inches='tight')


def unique_values(seq, index):
    seen = set()
    seen_add = seen.add
    return [str(x[index]) for x in seq if not (x[index] in seen or seen_add(x[index]))]


def age_state_cost(con):
    result = query_for_list(con, AGE_RATE_QUERY)
    rows = result['rows']
    desc = result['desc']
    states = unique_values(rows, 0)
    ages = unique_values(rows, 1)
    xbins = np.linspace(0, len(states), len(states)+1)
    ybins = np.linspace(0, len(ages), len(ages)+1)

    f = open('age_state_cost.txt', 'w')
    f.write('State\t\t\tAge\t\t\tAverage\n')
    query_info = []
    query_array_info = np.zeros([len(states), len(ages)])

    for row in rows:
        query_info.append({desc[0][0]: str(row[0]), desc[1][0]: str(row[1]), desc[2][0]: float(row[2])})
        query_array_info[states.index(str(row[0]))][ages.index(str(row[1]))] = float(row[2])
        f.write(str(row[0]) + '\t\t\t' + str(row[1]) + '\t\t\t' + str(float(row[2]))+'\n')

    f.close()

    plt.clf()
    font = {'size': 8}
    plt.rc('font', **font)

    plt.title('State-Age Rate Cost')
    plt.xticks(ybins, ages)
    plt.xticks(rotation=90)
    plt.xlabel('Age')
    plt.yticks(xbins, states)
    plt.ylabel('State')
    plt.imshow(query_array_info, interpolation='none')
    plt.colorbar(label='Annual Cost in Dollars ($)')
    pylab.savefig('AGE_STATE_COST.png', bbox_inches='tight')


# Main
def main():
    con = None
    try:
        con = lite.connect(DATABASE)
        state_benefits_cost_sharing(con)
        age_state_cost(con)
    except Exception as ex:
        print("Error %s:" % ex.args[0])
        sys.exit(1)
    finally:
        if con:
            con.close()

# Python's main
if __name__ == "__main__":
    main()
