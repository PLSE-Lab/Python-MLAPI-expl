import numpy as np
import csv
import sys
from fnmatch import fnmatch
from math import log
import matplotlib.pyplot as plt

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

def compute_entropy(data):
  n_survived = np.count_nonzero(data[:, 1] == '1')
  freq_survived = float(n_survived) / len(data)

  entropy = 0
  if freq_survived != 0.0:
    entropy -= freq_survived * log(freq_survived) / log(2)
  if freq_survived != 1.0:
    entropy -= (1 - freq_survived) * log(1 - freq_survived) / log(2)
  return entropy


def count_survived(data):
  part = 0.0
  for term in data:
    if term[SURVIVED] == '1':
      part += 1
  return part / len(data)


class decision_tree_node(object):
  def __init__(self, node_func=None, true_node=None, false_node=None):
    self.node_func = node_func
    self.true_node = true_node
    self.false_node = false_node
    self.data = np.array([])

  def fill(self, data_term):
    assert (self.node_func == None or
            self.true_node != None and self.false_node != None)

    if len(self.data) != 0:
      self.data = np.append(self.data, [data_term], 0)
    else:
      self.data = np.array([data_term])
    
    if self.node_func != None:
      if self.node_func(data_term):
        self.true_node.fill(data_term)
      else:
        self.false_node.fill(data_term)

  def predict(self, data_term):
    assert (self.node_func == None or
            self.true_node != None and self.false_node != None)

    if self.node_func != None:
      if self.node_func(data_term):
        return self.true_node.predict(data_term)
      else:
        return self.false_node.predict(data_term)
    else:
      if len(self.data) > min_n_votes:
          return '1' if count_survived(self.data) > 0.5 else '0'
      else:
        return ''

  def update_node(self, node_func, true_node, false_node):
    self.node_func = node_func
    self.true_node = true_node
    self.false_node = false_node
    assert (self.node_func == None or
            self.true_node != None and self.false_node != None)


def build_decision_tree(data, features):
  root = decision_tree_node()
  for term in data:
    root.fill(term)
  return build_decision_tree_node(root, features)


def build_decision_tree_node(node, features):
  data_inside_node = node.data
  best_decision = None
  best_decision_mean_entropy = compute_entropy(data_inside_node)
  
  for feature in features:
    feature_values = []
    for term in data_inside_node:
      if term[feature] not in feature_values:
        feature_values.append(term[feature])
    for value in feature_values:
      node.data = np.array([])
      true_node = decision_tree_node()
      false_node = decision_tree_node()
      node.update_node(lambda data_term: data_term[feature] == value,
                       true_node, false_node)
      for term in data_inside_node:
        node.fill(term)

      if len(true_node.data) != 0 and len(false_node.data) != 0: 
        true_node_entropy = compute_entropy(true_node.data)
        false_node_entropy = compute_entropy(false_node.data)
        mean_entropy = (true_node_entropy + false_node_entropy) / 2
        if mean_entropy < best_decision_mean_entropy:
          best_decision = [feature, value]
          best_decision_mean_entropy = mean_entropy

  if best_decision != None:
    node.data = np.array([])
    true_node = decision_tree_node()
    false_node = decision_tree_node()
    node.update_node(lambda data_term: data_term[best_decision[0]] == best_decision[1],
                     true_node, false_node)
    for term in data_inside_node:
      node.fill(term)
    if len(true_node.data) > 1:
      build_decision_tree_node(true_node, features)
    if len(false_node.data) > 1:
      build_decision_tree_node(false_node, features)
  else:
    node.update_node(None, None, None)
    node.data = np.array([])
    for term in data_inside_node:
      node.fill(term)
  return node


def make_intervals(values, max_n_bins, min_n_values):
  bins = []
  min_value = np.min(values)
  init_length = (np.max(values) - min_value) / (max_n_bins - 1)
  for i in range(max_n_bins):
    bins.append(min_value + i * init_length)
  assert(max_n_bins == len(bins))

  n_values = []
  for i in range(len(bins) + 1):
    n_values.append(0)

  for value in values:
    n_values[get_interval(bins, value)] += 1

  i = 0
  while i < len(n_values) - 1:
    if n_values[i] < min_n_values:
      del bins[i]
      n_values[i + 1] += n_values[i]
      del n_values[i]
    else:
      i += 1
  assert(len(n_values) == len(bins) + 1)
  return bins


def get_interval(bins, value):
  if value <= bins[0]:
    return 0
  if value > bins[-1]:
    return len(bins)

  for i in range(1, len(bins)):
    if bins[i - 1] < value and value <= bins[i]:
      return i


with open('../input/train.csv') as file:
  csv_file = csv.reader(file)
  header = csv_file.__next__()
  print('Header of train data: %s' % header)
  train_data = np.array([line for line in csv_file])
  print('First train sample: %s' % train_data[0])

with open('../input/test.csv') as file:
  csv_file = csv.reader(file)
  csv_file.__next__()  # Skip header.
  test_data = np.array([line for line in csv_file])
  test_data = np.insert(test_data, 1, '0', axis=1)  # Append Survived column.
  for term in test_data:
    term[NAME] = term[NAME].replace('\"', '')
  print('First test sample: %s' % test_data[0])


# Persons with 0 parch and sibsp.
min_n_votes = 3
features = [SEX, PCLASS, AGE, EMBARKED, FARE, CABIN, TICKET]
alone_persons = []
ages = []
fares = []

for term in train_data:
  if term[PARCH] == '0' and term[SIBSP] == '0':
    alone_persons.append(term)
alone_persons = np.array(alone_persons)
test_data_copy = np.copy(test_data)

# Collect ages and fares distribution.
for dataset in [train_data, test_data_copy]:
  for term in dataset:
    if term[AGE] != '':
      ages.append(float(term[AGE]))
    if term[FARE] != '':
      fares.append(float(term[FARE]))
      
# Save cabin character prefix.
for dataset in [alone_persons, test_data_copy]:
  for term in dataset:
    if term[CABIN] != '':
      for i in range(len(term[CABIN])):
        if '0' <= term[CABIN][i] and term[CABIN][i] <= '9':
          term[CABIN] = term[CABIN][:i]
          break
    else:
      term[CABIN] = 'none'
    
    # If ticket has only numbers - set length.
    if term[TICKET] != '':
      if not fnmatch(term[TICKET], '*[A-Z]*'):
        term[TICKET] = ('%d' % len(term[TICKET]))
      else:
        term[TICKET] = 'chars_prefix'
    else:
      term[TICKET] = 'none'

max_ages_bins = 10
ages_bins = make_intervals(ages, max_ages_bins, 10)

max_fares_bins = 10
fares_bins = make_intervals(fares, max_fares_bins, 10)

for dataset in [alone_persons, test_data_copy]:
  for term in dataset:
    if term[AGE] != '':
      interval = get_interval(ages_bins, float(term[AGE]))
      term[AGE] = ('%d' % interval)
    else:
      term[AGE] = 'none'

    if term[FARE] != '':
      interval = get_interval(fares_bins, float(term[FARE]))
      term[FARE] = ('%d' % interval)
    else:
      term[FARE] = 'none'

decision_tree = build_decision_tree(alone_persons, features)

predicted = 0.0
passed = 0.0
for term in alone_persons:
  if term[PARCH] == '0' and term[SIBSP] == '0':
    pred = decision_tree.predict(term)
    if pred != '':
      if term[SURVIVED] == pred:
        predicted += 1  
    else:
      passed += 1

print ('Desicion tree on train: %f %f' %
       (predicted / len(alone_persons), passed / len(alone_persons)))

tickets = {}
for term in train_data:
  ticket_id = term[TICKET]
  if ticket_id in tickets:
    tickets[ticket_id].append(term)
  else:
    tickets[ticket_id] = [term]

# Initial.
# males - 0
# females - 1
# females from 3rd class with fare > 20 - 0
for term in test_data:
  if term[SEX] == 'male':
    term[SURVIVED] = 0
  else:
    if term[PCLASS] == '3' and term[FARE] != '' and float(term[FARE]) > 20:
      term[SURVIVED] = 0
    else:
      term[SURVIVED] = 1

# Alone persons.
for term in test_data_copy:
  if term[PARCH] == '0' and term[SIBSP] == '0':
    predict = decision_tree.predict(term)
    if predict != '':
      term[SURVIVED] = predict 

# Family tickets.
# All test samples with existing train ticket (>1 in train set) where entropy=0
# predicts as all.
for term in test_data:
  ticket_id = term[TICKET]
  if ticket_id in tickets and len(tickets[ticket_id]) > 1:
    if compute_entropy(np.array(tickets[ticket_id])) == 0.0:
      term[SURVIVED] = tickets[ticket_id][0][SURVIVED]

# Hand crafted predictions.
predictions = {
  # Two parents family model
  'Mallet, Mrs. Albert (Antoinette Magnin)': '1',
  'Brown, Miss. Edith Eileen': '1',
  'Crosby, Mrs. Edward Gifford (Catherine Elizabeth Halstead)': '1',
  'Dodge, Mrs. Washington (Ruth Vidaver)': '1',
  'Danbom, Master. Gilbert Sigvard Emanuel': '0',
  'Drew, Master. Marshall Brines': '1',
  'Spedden, Master. Robert Douglas': '1',
  'Allison, Mr. Hudson Joshua Creighton': '1',
  'Dean, Mrs. Bertram (Eva Georgetta Light)': '1',  
  'Goodwin, Mr. Charles Frederick': '0',
  'Goodwin, Miss. Jessie Allis': '0',

  # Sinlge parent family model.
  'Aks, Master. Philip Frank': '1',
  'Palsson, Master. Paul Folke': '0',
  'Rice, Master. Albert': '0',
  'Greenfield, Mrs. Leo David (Blanche Strouse)': '1',
  'Williams, Mr. Richard Norris II': '1',
  'Becker, Mrs. Allen Oliver (Nellie E Baumgardner)': '1',
  'Olsen, Master. Artur Karl': '0',
  'Rosblom, Miss. Salli Helena': '0',
  'Touma, Miss. Maria Youssef': '1',
  'Touma, Master. Georges Youssef': '1',
  'Coutts, Mrs. William (Winnie Minnie Treanor)': '1',
  'Elias, Mr. Joseph': '0',
  'Quick, Miss. Winifred Vera': '1',
  'Cardeza, Mrs. James Warburton Martinez (Charlotte Wardle Drake)': '1',
  'Wells, Master. Ralph Lester': '1',
  'Wells, Mrs. Arthur Henry (Addie Dart Trevaskis)': '1',
  'Compton, Mr. Alexander Taylor Jr': '1',
  'Compton, Mrs. Alexander Taylor (Mary Eliza Ingersoll)': '1',

  # Hand crafted relations.
  'Moubarek, Mrs. George (Omine Amenia Alexander)': '1',
  'Sage, Miss. Ada': '0',
  'Sage, Master. William Henry': '0',
  'Sage, Mr. John George': '0',
  'Sage, Mrs. John (Annie Bullen)': '0',
  'Lefebre, Mrs. Frank (Frances)': '0',
  'Kink-Heilmann, Mrs. Anton (Luise Heilmann)': '1',  
  'Hirvonen, Mrs. Alexander (Helga E Lindqvist)': '1',
  'Davies, Mrs. John Morgan (Elizabeth Agnes Mary White) ': '1', 
  'Karun, Mr. Franz': '1',
  'Wells, Mrs. Arthur Henry (Addie Dart Trevaskis)': '1',
  'Wells, Master. Ralph Lester': '1',
  'Abbott, Master. Eugene Joseph': '1',
  'Nakid, Mrs. Said (Waika Mary Mowad)': '1',
}

for name in predictions:
  found = False
  for term in test_data:
    if term[NAME] == name:
      term[SURVIVED] = predictions[name]
      found = True
      break
  if not found:
    print ('%s not found' % name)

with open('./decision_tree_using_tickets_length_v2.csv', 'w') as file:
  csv_file = csv.writer(file)
  csv_file.writerow(header[:2])
  for row in test_data:
    csv_file.writerow(row[:2])
