import numpy as np

print ("Hello kaggle! Oh wow, Python 3...")

print ("numpy? yes.")

popfile = "../input/pums/ss13pusa.csv"

print ("Opening" + popfile)

lines = 0
for line in open(popfile):
    print (line)
    lines += 1
    if (lines > 10):
        break
#popa_data = np.genfromtxt(popfile)

#for item in popa_data[0]:
#    print (item)
    
print ("Complete!")