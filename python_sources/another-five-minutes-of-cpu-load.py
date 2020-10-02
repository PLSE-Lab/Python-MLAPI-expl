from time import time 

t1 = time()
while True:
    x = 2*2
    if time() - t1 > 5*60:
        break
