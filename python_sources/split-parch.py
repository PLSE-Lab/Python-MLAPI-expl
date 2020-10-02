# This sript split PARCH field to PARENTS and CHILDREN

import numpy as np
import csv
import sys
from fnmatch import fnmatch

ID = 0
SURVIVED = 1
PCLASS = 2
NAME = 3
SEX = 4
AGE = 5
SIBSP = 6
PARCH = 7
TICKET = 8
FARE = 9
CABIN = 10
EMBARKED = 11

data = []
with open('../input/train.csv') as file:
    csv_file = csv.reader(file)
    csv_file.__next__()  # Skip header.
    data = [line for line in csv_file]

with open('../input/test.csv') as file:
    csv_file = csv.reader(file)
    csv_file.__next__()  # Skip header.
    test_data = [line for line in csv_file]
    test_data = np.insert(test_data, 1, '?', axis=1)  # Append Survived column.
    for term in test_data:
        data.append(term)
        
single_parent_families = open('./single_parent_families.txt', 'w')
two_parents_families = open('./two_parents_families.txt', 'w')

data = np.array(data)
print ('Total number of persons: %d' % len(data))

# Initially save persons with PARCH != 0.
data = data[data[:,PARCH] != '0']
print ('Total number of parents/children: %d' % len(data))

# 1. Group persons by surname (first substring in name before ',').
surname_groups = {}
for term in data:
    surname = term[NAME][:term[NAME].find(',')]
    if surname in surname_groups:
        surname_groups[surname].append(term)
    else:
        surname_groups[surname] = [term]
print ('Number of surname groups: %d' % len(surname_groups))


# Preprocess groups with single persons.
removed_keys = []
for group in surname_groups:
    if len(surname_groups[group]) == 1:
        person = surname_groups[group][0]
        candidates = []
        for surname in surname_groups:
            if surname != group and \
              len(surname_groups[surname]) != 0 and \
              (fnmatch(person[NAME], '*(*%s*)*' % surname)):
                candidates.append(surname)
        if len(candidates) == 1:
            surname_groups[candidates[0]].append(person)
            surname_groups[group] = []
            removed_keys.append(group)
        else:
            assert(len(candidates) == 0)
for key in removed_keys:
    surname_groups.pop(key)
print ('Number of accumulated surname groups: %d' % len(surname_groups))

# for group in surname_groups:
#     if len(surname_groups[group]) == 1:
#         print (surname_groups[group][0])
        
# sys.exit()

# key - id, value - [number of parents, number of children]
processed_persons = {}

# 2. Find trivial family patterns:
# 2.1. Single parent and children
#      parent conditions:
#      * PARCH: number of members - 1
#      * NAME: Mr. if male or Mrs. if female
#      * older than children
#      childred conditions:
#      * PARCH: 1
#      * SIBSP: number of members - 2
removed_keys = []
for group in surname_groups:
    family = surname_groups[group]
    n_members = len(family)
    candidates = []
    for term in family:
        if int(term[PARCH]) == n_members - 1 and \
           ('Mr.' in term[NAME] or 'Capt.' in term[NAME] or 'Mrs.' in term[NAME]):
            candidates.append(term)
    
    if len(candidates) >= 1:
        if len(candidates) > 1:
            # Oldest member as parent.
            for term in candidates:
                assert(term[AGE] != '')
            candidates = sorted(candidates, key=lambda term: float(term[AGE]), reverse=True)
                
        parent = candidates[0]
        append_family = True
        for term in family:
            if term[ID] != parent[ID] and \
               (int(term[PARCH]) != 1 or \
                int(term[SIBSP]) != n_members - 2 or \
                term[AGE] != '' and parent[AGE] != '' and float(term[AGE]) >= float(parent[AGE])):
                append_family = False
                break
        if append_family:
            removed_keys.append(group)
            single_parent_families.write(parent[ID])
            for term in family:
                assert (term[ID] not in processed_persons)
                if term[ID] != parent[ID]:
                    processed_persons[term[ID]] = [1, 0]
                    single_parent_families.write(' ' + term[ID])
                else:
                    processed_persons[term[ID]] = [0, n_members - 1]
            single_parent_families.write('\n')


print ('Number of families with single parent: %d' % len(removed_keys))
for key in removed_keys:
    surname_groups.pop(key)
    
# 2.2. Two parents and children
#      parents conditions:
#      * PARCH: number of members - 2
#      * SIBSP: 1
#      * NAME: Mr. if male and Mrs. if female
#      * older than children
#      childred conditions:
#      * PARCH: 2
#      * SIBSP: number of members - 3
removed_keys = []
for group in surname_groups:
    family = surname_groups[group]
    n_members = len(family)
    mr_candidates = []
    mrs_candidates = []
    for term in family:
        if int(term[PARCH]) == n_members - 2 and \
          int(term[SIBSP]) == 1:
            if 'Mr.' in term[NAME] or \
               'Capt.' in term[NAME] or \
               'Rev.' in term[NAME] or \
               'Dr.' in term[NAME] or \
               'Rev.' in term[NAME]:
                mr_candidates.append(term)
            elif 'Mrs.' in term[NAME]:
                mrs_candidates.append(term)
    if len(mr_candidates) == 1 and len(mrs_candidates) == 1:
        parents = [mr_candidates[0], mrs_candidates[0]]
        append_family = True
        for term in family:
            if term[ID] != parents[0][ID] and term[ID] != parents[1][ID] and \
               (int(term[PARCH]) != 2 or \
                int(term[SIBSP]) != n_members - 3 or \
                term[AGE] != '' and parents[0][AGE] != '' and float(term[AGE]) >= float(parents[0][AGE]) or \
                term[AGE] != '' and parents[1][AGE] != '' and float(term[AGE]) >= float(parents[1][AGE])):
                append_family = False
                break
        if append_family:
            removed_keys.append(group)
            two_parents_families.write(parents[0][ID] + ' ' + parents[1][ID])
            for term in family:
                assert (term[ID] not in processed_persons)
                if term[ID] != parents[0][ID] and term[ID] != parents[1][ID]:
                    processed_persons[term[ID]] = [2, 0]
                    two_parents_families.write(' ' + term[ID])
                else:
                    processed_persons[term[ID]] = [0, n_members - 2]
            two_parents_families.write('\n')
                
print ('Number of families with two parents: %d' % len(removed_keys))
for key in removed_keys:
    surname_groups.pop(key)

print ('%d surname groups left' % len(surname_groups))

# [parents id], [{children ids}]
hand_crafted_relations = [
    [['775'], ['530', '944', '438']],
    [['438'], ['408', '832']],
    [['1057', '1286'], ['185']],
    [['1133'], ['581', '601']],
    [['821', '1200'], ['984']],
    [['588', '1289'], ['540']],
    [['849'], ['721']],
    [['1222'], ['550', '146']],
    [['249', '872'], ['137']],
    [['780'], ['690']],
    [['660'], ['394', '216']],
    [['737'], ['148', '437', '1059', '87']],
    [['300'], ['119', '1076']],
    [['313', '1041'], ['418']],
    [['248'], ['756', '1130']],
    [['14', '611'], ['69', '543', '542', '814', '851', '120']],
]

PARENTS = 0
CHILDREN = 1
for rel in hand_crafted_relations:
    for parent in rel[PARENTS]:
        if parent not in processed_persons:
            processed_persons[parent] = [0, 0]
        processed_persons[parent][CHILDREN] = len(rel[CHILDREN])
        
    for child in rel[CHILDREN]:
        if child not in processed_persons:
            processed_persons[child] = [0, 0]
        processed_persons[child][PARENTS] = len(rel[PARENTS])
    
    if len(rel[PARENTS]) == 1:
        single_parent_families.write(rel[PARENTS][0])
        for child in rel[CHILDREN]:
            single_parent_families.write(' ' + child)
        single_parent_families.write('\n')
    else:
        two_parents_families.write(rel[PARENTS][0] + ' ' + rel[PARENTS][1])
        for child in rel[CHILDREN]:
            two_parents_families.write(' ' + child)
        two_parents_families.write('\n')

outpaths = ['./train_modif.csv', './test_modif.csv']
for i, path in enumerate(['../input/train.csv', '../input/test.csv']):
    with open(path) as file:
        data = []
        csv_file = csv.reader(file)
        header = csv_file.__next__()  # Skip header.
        data = np.array([line for line in csv_file])

        PARENTS = len(header)
        CHILDREN = PARENTS + 1
        
        header = np.insert(header, PARENTS, 'Parents', axis=0)
        header = np.insert(header, CHILDREN, 'Children', axis=0)
        data = np.insert(data, PARENTS, '0', axis=1)
        data = np.insert(data, CHILDREN, '0', axis=1)
        
        for term in data:
            if term[ID] in processed_persons:
                term[PARENTS] = processed_persons[term[ID]][0]
                term[CHILDREN] = processed_persons[term[ID]][1]
        with open(outpaths[i], 'w') as outfile:
            csv_file = csv.writer(outfile)
            csv_file.writerow(header)
            for term in data:
                csv_file.writerow(term)
                
single_parent_families.close()
two_parents_families.close()

