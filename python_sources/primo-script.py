# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import model_selection as modsel, cluster as cl, decomposition as dec, preprocessing as prep
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

"""from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))"""

# Any results you write to the current directory are saved as output.
people = pd.read_csv("../input/HR_comma_sep.csv")

# Detach the binary variable "left" (whether the employee left) from the main table
people_left = people["left"]
people = people.drop("left",1)

# Transform salary categorization into rank
people["salary"][people["salary"]=="low"] = 0
people["salary"][people["salary"]=="medium"] = 1
people["salary"][people["salary"]=="high"] = 2

# Leaving alone the department category for the moment
people_nodept = people.drop("sales",1)

# Trying to have an informative 3d visualization, so setting PCA components to 3 for the moment
people_s = prep.scale(people_nodept)
people_pca =dec.PCA(n_components=3,svd_solver='full').fit(people_s).transform(people_s)

# Appending the "left" bit to the transformed matrix
people_pca = np.append(people_pca,people_left.values.reshape((-1,1)),axis=1)

# Creating two matrixes, one with all employees that left and another with all employees that stayed.
# Note that np.where creates a set of tuples, so it is necessary to retrieve the resulting filtered matrix with the index 0.
people_pca_left = people_pca[np.where(people_pca[:,3]==1),:][0]
people_pca_stay = people_pca[np.where(people_pca[:,3]==0),:][0]

# Creating the 3d plot with the two subsets of people_pca
fig = plt.figure()
scatter = fig.add_subplot(111,projection="3d")

# People that left are plotted with red dots, people that stayed with black ones
scatter.scatter(people_pca_left[:,0],people_pca_left[:,1],people_pca_left[:,2],c="red")
scatter.scatter(people_pca_stay[:,0],people_pca_stay[:,1],people_pca_stay[:,2],c="black")
plt.show()
fig.savefig("pca.png")