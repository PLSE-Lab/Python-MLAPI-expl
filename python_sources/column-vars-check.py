# Competition "stat331w11" had two different teams finish in the same spot (307)
# spot 308 is missing. Perfect tie?

import csv
from collections import defaultdict, Counter

data = defaultdict(list)
for e, r in enumerate(csv.DictReader(open("../input/Teams.csv"))):
    data[r["CompetitionId"]].append( r["Ranking"] )

for d in data:
    if len(data[d]) != len(set(data[d])):
        print(d,sorted([int(f) for f in data[d]]))
        print(Counter(data[d]))
        
for e, r in enumerate(csv.DictReader(open("../input/Competitions.csv"))):
    if r["Id"] == "2555":
        print(r["CompetitionName"],r["Title"])