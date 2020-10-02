import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

data = pd.read_csv('../input/HR_comma_sep.csv');

print(data.describe());

figure = plt.figure(figsize=(5,6))
axe = figure.add_subplot(1,1,1)
axe.scatter(data['satisfaction_level'],data['last_evaluation']);
plt.show();