import os

dirname="images"
os.mkdir(dirname)
for i in range(1050):
    f = open(dirname + "/" + str(i), "w")
    f.writelines(["hello", "world"])
    f.close()

f = open("another_file", "w")
f.writelines(["1"])
f.close()