from time import time 

t1 = time()
while True:
    x = 2*2
    if time() - t1 > 4*60:
        break
