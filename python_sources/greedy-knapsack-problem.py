def GreedyKnapsack(m,n):
    for i in range(0,n):
        global x
        x=[]
        x.insert(i,0.0)
        u=m
        for i in range(0,n):
            if(weight[i]>u):
                break
            x.insert(i,1.0)
            u=u-weight[i]
        if(i<n):
            q=u/weight[i]
            x.insert(i,q)

n=int(input("Enter the number of items : "))

print("Enter items : ")

profit=list(map(float,input("\tEnter profit for items: ").split()))

weight=list(map(float,input("\tEnter weight for items: ").split()))

density=[]
for i in range(0,n):
    z=profit[i]/weight[i]
    density.append(z)

for i in range(0,n,++i):
    for j in range(i+1,n,++j):
        if(density[i] < density[j]):
            a = density[i]
            density[i] = density[j]
            density[j] = a              
            a = weight[i]
            weight[i] = weight[j]
            weight[j] = a
            a = profit[i]
            profit[i] = profit[j]
            profit[j] = a 

m=float(input("\nENTER MAX WEIGHT ALLOWED: "))

print("\nAfter arranging in decreasing order: ")
for i in range(0,n):
    print("For item no. ",i+1,", Density is: ",density[i]," and Weight is: ",weight[i],"\n" )

GreedyKnapsack(m,n)

maxprofit=0.0
for i in range(0,n):
    print("Usage of item ",i+1," is : ",x[i]," with Weight usage = ",weight[i]*x[i])
    maxprofit=maxprofit+(profit[i]*x[i])
print(" The Max profit using Greedy Algo comes out to be : ",maxprofit)
