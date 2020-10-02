words=[]
with open("../input/w3.dll") as f:
    data=f.read()
words=data.split("|")
#print(words)
for word in words:
    print(word)