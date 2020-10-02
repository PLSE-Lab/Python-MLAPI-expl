import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

x = np.linspace(0,1)

for i in range(3):
    y = i*x
    plt.scatter(x, y)
    plt.ylim([0,5])
    plt.savefig('image%d.png' % i)
    plt.clf()   