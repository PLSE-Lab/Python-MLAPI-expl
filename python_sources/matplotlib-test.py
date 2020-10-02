import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.fill_between([0, 1], [0, 1])
#plt.plot(np.linspace(0,10,50), np.sin(np.linspace(0,10,50)))
plt.savefig('test1.png')
print(matplotlib.__version__)
