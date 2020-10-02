import json
import csv

from collections import defaultdict


states = [
    "Andhra Pradesh",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
    "Chhattisgarh",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Jammu and Kashmir",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Madhya Pradesh",
    "Maharashtra", # to accomodate both Maharastra and Maharashtra 
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Odisha",
    "Punjab",
    "Rajasthan",
    "Sikkim",
    "Tamil Nadu",  # To accomodate both Tamilnadu and Tamil Nadu
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
    "West Bengal",
    "Andaman and Nicobar",
    "Chandigarh",
    "Dadra and Nagar Haveli",
    "Daman and Diu",
    "National Capital Territory of Delhi",
    "Lakshadweep",
    "Puducherry"
]

with open("../input/visits.json", "r") as f:
    visits = json.load(f)

assert len(states) == 29+7

counter = defaultdict(int)

for visit in visits:
    for state in states:
        if state in visit["places"]:
            counter[state] += 1
    if "Tamilnadu" in visit["places"]:
        counter["Tamil Nadu"] += 1
    if "Maharastra" in visit["places"]:
        counter["Maharashtra"] += 1

csvfile = open("state_wise_count.csv", "w") 
writer = csv.writer(csvfile)
writer.writerow(["State", "Visits"])

for state, visit_count in counter.items():
    writer.writerow([state, visit_count])

csvfile.close()