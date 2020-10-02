import io
import zipfile
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt

row_counts = [1613672,1519123]
bins = np.zeros(9)

print("Start of dataset 1")
for fNr in range(2):
    if (fNr == 0):
        alpha = 'a'
    else:
        alpha = 'b'
    print("Start of dataset {0}".format(alpha))
     
    csvf = csv.reader(open('../input/pums/ss13pus{0}.csv'.format(alpha), 'rU'))
    header = csvf.__next__()

    pumaColNr = header.index('PUMA')
    stColNr = header.index('ST')

    colNr = header.index('PINCP')


    for i in range(row_counts[fNr]):
        row=csvf.__next__()
        puma=row[pumaColNr]
        state=row[stColNr]
        col=row[colNr]
        if (col == '' or int(col) == 0):
            continue

        # get the first digit of the income (without the leading zeros)
        colStr=str(int(col))
        if (int(col) > 0):
            firstDigit = int(colStr[0])
        else:
            firstDigit = int(colStr[1])

        bins[firstDigit-1] += 1
    print("End of dataset {0}".format(alpha))

benford_bins = [np.log10(1+1.0/d) for d in range(1,10)]
bins /= sum(bins)

N = 9
ind = np.arange(N)  # the x locations for the groups
width = 0.4       # the width of the bars

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)
ax.set_title("Benford's law: U.S. Income", fontsize=20)


rects1 = ax.bar(ind, bins, width, color='r')
rects2 = ax.bar(ind+width, benford_bins, width, color='b')
ax.set_xticks(ind+width)
ax.set_xticklabels( [str(int(i)) for i in range(1,10)] )
ax.set_yticklabels( [str(int(i)*5)+"%" for i in range(8)] )


ax.legend( (rects1[0], rects2[0]),
('First digit of income', "Benford's distribution") )

plt.savefig('benford.png')
