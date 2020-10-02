import multiprocessing
print(multiprocessing.cpu_count())
from time import time
t1 = time()
while time() - t1 < 20:
    x = 2
