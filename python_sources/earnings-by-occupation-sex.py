import io
import zipfile
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
import collections

row_counts = [1613672,1519123]

occpNr2Gr = {430: "Management",950: "Business",1240: "IT",1560: "Engineering",1965: "Science",2060: "Community and Social Service",2160: "Legal",2250: "Education",2960: "Arts & Sports",3540: "Healthcare",3655: "Healthcare Support",3955: "Protective Service",4160: "Food",4250: "Building Cleaning",4650: "Personal Care",4965: "Sales",5940: "Office and Administrative Support",6130: "Farming, Fishing",6940: "Construction and Extraction",7630: "Installation, Maintenance, and Repair",8965: "Production",9240: "Transportation",9750: "Material Moving",9830: "Military"}

occpNr2Gr = collections.OrderedDict(sorted(occpNr2Gr.items()))

occpNumbers = list(occpNr2Gr.keys())


occp2Ind = {"Management": 0, "Business": 1, "IT": 2, "Engineering": 3, "Science": 4, "Community and Social Service": 5, "Legal": 6, "Education": 7, "Arts & Sports": 8, "Healthcare": 9, "Healthcare Support": 10,"Protective Service": 11, "Food": 12,
"Building Cleaning": 13, "Personal Care": 14, "Sales": 15, "Office and Administrative Support": 16, "Farming, Fishing": 17, "Construction and Extraction": 18, "Installation, Maintenance, and Repair": 19, "Production": 20,"Transportation": 21, "Material Moving": 22, "Military": 23}

occp2cuInd = np.zeros(48)

data = np.zeros((48,253070))

for fNr in range(2):
    if (fNr == 0):
        alpha = 'a'
    else:
        alpha = 'b'

    csvf = csv.reader(open('../input/pums/ss13pus{0}.csv'.format(alpha), 'rU'))
    header = csvf.__next__()

    pumaColNr = header.index('PUMA')
    stColNr = header.index('ST')
    print(stColNr)

    earnNr = header.index('PERNP')
    print(header[earnNr])
    occpNr = header.index('OCCP')
    print(header[occpNr])
    sexNr = header.index('SEX')
    print(header[sexNr])

    for i in range(row_counts[fNr]):
        row=csvf.__next__()
        puma=row[pumaColNr]
        state=row[stColNr]
        earnCol=row[earnNr]
        occpCol=row[occpNr]
        sexCol=row[sexNr]

        if (earnCol == '' or occpCol == ''):
            continue

        earnCol  = int(earnCol)
        occpCol = int(occpCol)
        sexCol = int(sexCol)


        for coc in occpNumbers:
            if (occpCol <= coc):
                currentOCCP = occpNr2Gr[coc]
                if (sexCol == 2): # Female
                    rowIndOCCP = occp2Ind[currentOCCP]
                else: # Male
                    rowIndOCCP = occp2Ind[currentOCCP]+24

                data[rowIndOCCP][occp2cuInd[rowIndOCCP]] = earnCol
                occp2cuInd[rowIndOCCP] += 1
                break

print("data: ",occp2cuInd)
print("max: ",np.max(occp2cuInd))

ind2OCCP = np.array(["Management","Business", "IT", "Engineering", "Science",
"Community\nSocial Service", "Legal", "Education", "Arts & Sports", "Healthcare", "Healthcare Support","Protective Service", "Food",
"Building Cleaning", "Personal Care", "Sales", "Office and Administrative Support", "Farming, Fishing", "Construction and Extraction", "Installation, Maintenance, and Repair", "Production","Transportation", "Material Moving", "Military"])

meansFemale = np.zeros(24)
nonZerosFemale = np.zeros(24)
for i in range(24):
    nonZerosFemale[i] = np.count_nonzero(data[i])
    meansFemale[i] = np.mean(data[i][:nonZerosFemale[i]])
print(meansFemale)

meansMale = np.zeros(24)
nonZerosMale = np.zeros(24)
for i in range(24,48):
    nonZerosMale[i-24] = np.count_nonzero(data[i])
    meansMale[i-24] = np.mean(data[i][:nonZerosMale[i-24]])

# sort the occupations and update all arrays
meansInd = np.argsort(meansFemale)

meansFemale = meansFemale[meansInd]
meansMale = meansMale[meansInd]
ind2OCCP = ind2OCCP[meansInd]

nonZerosMale = nonZerosMale[meansInd]
nonZerosFemale = nonZerosFemale[meansInd]

print("Female")
print(meansFemale)
print("Male")
print(meansMale)


N = 24
ind = np.arange(N)  # the x locations for the groups
width = 0.4     # the width of the bars

fig = plt.figure(figsize=(25,10))
ax = fig.add_subplot(111)
ax.set_title("Earnings in $ by Occupation & Sex", fontsize=20)


rects = []
rects.append(ax.barh(ind, meansFemale, width, color='r'))
rects.append(ax.barh(ind+width, meansMale, width, color='b'))

ax.set_yticks(ind+width)
ax.set_yticklabels( ind2OCCP)

ax.legend( (rects[0][0], rects[1][0]),
('Female', "Male") )


for bar,sexChar in zip(rects,["F","M"]):
    # Lastly, write in the ranking inside each bar to aid in interpretation
    c = 0
    for rect in bar:
        width = int(rect.get_width())

        if (sexChar == "F"):
            noPersons = (nonZerosFemale[c]/(nonZerosFemale[c]+nonZerosMale[c]))*100
        else:
            noPersons = (nonZerosMale[c]/(nonZerosFemale[c]+nonZerosMale[c]))*100

        barStr = format(noPersons, '.1f')+"%"
        xloc = 0.98*width  # Shift the text to the left side of the right edge
        clr = 'white'      # White on magenta
        align = 'right'

        # Center the text vertically in the bar
        yloc = rect.get_y()+rect.get_height()/2.0
        ax.text(xloc, yloc, barStr, horizontalalignment=align,
                verticalalignment='center', color=clr, weight='bold',fontsize=10)

        c += 1

plt.figtext(.2, .92, "The values inside the bars represent the distribution between the sexes for the given occupation")
plt.tight_layout()
plt.savefig('occupation-earnings-sex.png')

