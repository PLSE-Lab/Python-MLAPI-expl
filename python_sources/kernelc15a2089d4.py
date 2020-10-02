import numpy as np
import csv

with open('../input/train.csv', 'r') as f:
    reader = csv.reader(f)
    arr = []
    f = True
    for row in reader:
        if f:
            f = False
            continue
        arr.append(list(map(int, row[1][1:].split(' '))))

days = []
for person in arr:
    p_days = []
    for day in person:
        p_days.append((day-1) % 7 + 1)
    days.append(p_days)

with open('solution.csv', 'w') as f:
    f.write('id,nextvisit\n')
    for idx, person in enumerate(days):
        if arr[idx][-1] < 970:
            f.write(str(idx+1) + ', 0\n')
        else:
            values, counts = np.unique(person[10:], return_counts=True)
            f.write(str(idx+1) + ', ' + str(values[np.argmax(counts)]) + '\n')
