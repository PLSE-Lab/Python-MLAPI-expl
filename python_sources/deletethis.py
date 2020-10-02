#1.Find max of numbers in an array

#an array of numbers
a=[1,44,1000,6778,2223,676]
print(type(a))

max=0
for i in a :
    if max-i <=0:
        max=i        
    else:
        max=max
        
print(max)


#2. Sorting an array of numbers (ascending)
b=[1,44,1000,6778,-27,2223,0,676]

n=len(b)
print(len(b))

for i in range(0,len(b)):
    for j in range(i+1,len(b)):
        if b[i] > b[j]:
            b[i],b[j]=b[j],b[i]
        else:
            b[i],b[j]= b[i],b[j]
        
print(b)


