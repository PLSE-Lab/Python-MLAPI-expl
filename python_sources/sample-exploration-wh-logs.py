# Imports
import json

# Number of files to iterate through.
NO_FILES = 6

visitors = {}

# Iterate through each file.
for i in range(0, NO_FILES):
    f = open("../input/whlogs_%s.csv" % (str(i + 1)), "r")
    rows = f.readlines()

    for j in range(1, len(rows)):
        elements = rows[j].split(",")
        visitor = (elements[1].strip() + " " + elements[0].strip()).title()

        if visitor not in visitors:
            visitors[visitor] = 0
        
        visitors[visitor] += 1

    f.close()

f = open("results.json", "w")
f.write(json.dumps(visitors, indent=4, sort_keys=False))
f.close()
