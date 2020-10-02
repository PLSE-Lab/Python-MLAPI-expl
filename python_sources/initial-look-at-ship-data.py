#!/usr/bin/python3
# coding: utf-8
# kaggle_init_ship_data.py

import os
import numpy as np
import pandas as pd

ship_data = pd.read_csv(os.path.join('..', 'input', 'CLIWOC15.csv'), nrows=2000000)

print(str(len(ship_data.ShipName)) + ' records loaded')
print(str(len(set(ship_data.ShipName))) + ' ShipNames: ' + str(list(set(ship_data.ShipName))[0:5]))


print(str(len(set(ship_data.Year))) + ' years: ' + str(sorted(list(set(ship_data.Year)))))
