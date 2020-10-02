import csv
from collections import defaultdict

d = defaultdict(lambda: defaultdict(list))
model = defaultdict(dict)
for e, line in enumerate(open("../input/train.csv")):
	if e > 0:
		r = line.strip().split(",")
		x = r[1:-1]
		y = r[-1]
		for i in range(1932):
			d[i][x[i]].append(float(y))

	if e > 60000:
		break

for col in d:
	for var in d[col]:
		l = d[col][var]
		model[col][var] = sum(l)/float(len(l))

with open("sub.csv","wb") as outfile:		
	out = "ID,target\n"		
	for e, line in enumerate(open("../input/test.csv")):
		if e > 0:
			r = line.strip().split(",")
			x = r[1:]
			id = r[0]
			p = []
			for i in range(1932):
				try:
					p.append(model[i][x[i]])
				except:
					c=1
			out += "%s,%s\n"%(id,sum(p)/float(len(p)))
	outfile.write(out.encode("utf-8"))
