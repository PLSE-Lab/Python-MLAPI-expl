import pandas as pd
import numpy as np
from os.path import isfile
import math
if isfile("./train.csv"):
    data = pd.read_csv("./train.csv")
    visits = np.array([np.array(list(map(int, elem.split()))) for elem in data["visits"]])
    week_days = np.array([(elem - 1) % 7 for elem in visits])
    pauses = np.array([elem[1:] - elem[:-1] for elem in visits])
    
    result = []
    for v, d, p in zip(visits, week_days, pauses):
      if np.max(p) == p[-1] and np.mean(p) + np.std(p) * 3 < p[-1]:
        result.append(' 0')
        continue
      day_prob = [0.0] * 7
      for x in d:
        day_prob[x] += 1.0
      day_prob = [x / 157.0 for x in day_prob]
      mean_p = int(np.mean(p))
      max_prob = 0.0
      max_d = 1
      old_prob = 1.0
      max_p = np.max(p)
      for i in range(7 + max(mean_p - p[-1], 0)):
        p_prob = 1.0 - max(mean_p - p[-1] - i, 0) * 1.0 / max_p
        cur_prob = old_prob * day_prob[i % 7] * p_prob
        if cur_prob > max_prob:
          max_prob = cur_prob
          max_day = i % 7
        old_prob *= 1.0 - cur_prob
      result.append(' ' + str(max_day + 1))
    
    result_data = pd.DataFrame({'id': data['id'], 'nextvisit': result})
    result_data.to_csv("solution.csv", index=False, sep=',')
