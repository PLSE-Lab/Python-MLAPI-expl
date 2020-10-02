import pandas as pd
from pandas import read_csv

data = read_csv('../input/train.csv')
with open('solution.csv', 'w') as file:
    file.write('id,nextvisit\n')
    for i in data.iterrows():
        visits = i[1]['visits'].split()
        imp = [0] * 8
        for day in visits:
            imp[1 + (int(day) - 1) % 7] += int(day) // 7
        next_visit = imp.index(max(imp))
        file.write(str(i[1]['id']) + ', ' + str(next_visit) + '\n')
file.close()