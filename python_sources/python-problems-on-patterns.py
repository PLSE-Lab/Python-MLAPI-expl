#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Notebook consists of problems on patterns which uses loops.


# In[ ]:


# Pattern 1:
x = 5
for i in range(1,int(x)+1):
    print(i*"*")


# In[ ]:


# Pattern 2:
x = 5
y = int(x)
for i in range(1,y+1):
    print((y-i)*" ",i*"*")


# In[ ]:


# Pattern 3:
x = 5
y = int(x)
for i in range(1,y+1):
    print((6-i)*"*")


# In[ ]:


# Pattern 4:
x = 5
y = int(x)
for i in range(1,y+1):
    print((i-1)*" ",(6-i)*"*")


# In[ ]:


# Pattern 5:
x = 5
y = int(x)
for i in range(1,y+1):
    print((y-i)*" ",(2*i-1)*"*")


# In[ ]:


# Pattern 6:
x = 5
y = int(x)
for i in range(1,y+1):
    print((i-1)*" ",(2*(y-i)+1)*"*")


# In[ ]:


# Pattern 7:
x = 5
y = int(x)
for i in range(1,y):
    print((y-i)*" ",(2*i-1)*"*")
for j in range(1,y+1):
    print((j-1)*" ",(2*(y-j)+1)*"*")


# In[ ]:


# Pattern 8:
x = 5
y = int(x)
for i in range(1,y):
    print(i*"*")
for j in range(1,y+1):
    print((y+1-j)*"*")


# In[ ]:


# Pattern 9:
x = 5
y = int(x)
for i in range(1,y):
    print((y-i)*" ",i*"*")
for j in range(1,y+1):
    print((j-1)*" ",(y+1-j)*"*")


# In[ ]:


# Pattern 10:
x = 5
y = int(x)
for i in range(1,y+1):
    print((y-i)*" ",y*"*")


# In[ ]:


# Pattern 11:
x = 5
y = int(x)
for i in range(1,y+1):
    print((i-1)*" ",y*"*")


# In[ ]:


# Pattern 12:
x = 5
y = int(x)
for i in range(1,y):
    print((y+1-i)*"*")
for j in range(1,y+1):
    print(j*"*")


# In[ ]:


# Pattern 13:
x = 5
y = int(x)
for i in range(1,y):
    print((i-1)*" ",(y+1-i)*"*")
for j in range(1,y+1):
    print((y-j)*" ",j*"*")


# In[ ]:


# Pattern 14:
x = 5
y = int(x)
for i in range(1,y):
    print((i-1)*" ",(y+1-i)*"* ")
for j in range(1,y+1):
    print((y-j)*" ",j*"* ")


# In[ ]:


# Pattern 15:
x = 5
y = int(x)
print("*")
for i in range(2,y):
    print("*",end="")
    print((i-2)*" ",end="")
    print("*")
print(y*"*")


# In[ ]:


# Pattern 16:
x = 5
y = int(x)
print((y-2)*" ","*")
for i in range(2,y):
    print((y-i)*" ",end="")
    print("*",end="")
    print((i-2)*" ",end="")
    print("*")
print(y*"*")


# In[ ]:


# Pattern 17:
x = 5
y = int(x)
for i in range(1,y+1):
    for j in range(1,i):
        print(j,end="")
    print(i)


# In[ ]:


# Pattern 18:
x = 5
y = int(x)
for i in range(1,y+1):
    print(i*str(i))


# In[ ]:


# Pattern 19:
x = 5
y = int(x)
for i in range(1,y):
    for j in range (1,i):
        print(j,end="")
    print(i)
for k in range(1,y+1):
    for l in range(1,(y+1-k)):
        print(l,end="")
    print(y+1-k)


# In[ ]:


# Pattern 20:
x = 5
y = int(x)
for i in range(1,y+1):
    for j in range(1,y+1-i):
        print(j,end="")
    print(y+1-i)
for k in range(1,y+1):
    for l in range(1,k):
        print(l,end="")
    print(k)


# In[ ]:


# Pattern 21:
x = 5
y = int(x)
L1 = list(range(1,y+1))
L2 = list(reversed(L1))
for i in L1:
    for j in range(1,y+2-i):
        print((y+2-i-j),end="")
    print("")
for k in L2:
    for l in range(1,y+2-k):
        print((y+2-k-l),end="")
    print("")


# In[ ]:


# Pattern 22:
x = 5
y = int(x)
for i in range(1,y+1):
    print(" "*(y-i),end="")
    for j in range(1,i+1):
        print(j," ",end="")
    print("")


# In[ ]:


# Pattern 23:
x = 5
y = int(x)
for i in range(1,y+1):
    for j in range(1,y+2-i):
        print((y+1-j),end="")
    print("")


# In[ ]:


# Pattern 24:
x = 5
y = int(x)
for i in range(1,y+1):
    for j in range(1,i+1):
        print(y+1-j,end="")
    print("")


# In[ ]:


# Pattern 25:
x = 5
y = int(x)
for i in range(1,y+1):
    for j in range(1,y+2-i):
        print(j,end="")
    print("")


# In[ ]:


# Pattern 26:
x = 5
y = int(x)
k=1
for i in range(1,y+1):
    for j in range(1,i+1):
        print(k,end=" ")
        k=k+1
    print("")


# In[ ]:


# Pattern 27:
x = 5
y = int(x)
L1 = list(range(1,y+1))
L2 = list(reversed(L1))
for k in L2:
    for l in range(1,y+2-k):
        print((y+2-k-l),end="")
    print("")


# In[ ]:


# Pattern 28:
x = 5
y = int(x)
for i in range(1,y+1):
    for j in range(1,i+1):
        print(i+((j-1)*y),end=" ")
    print("")


# In[ ]:


# Pattern 29:
x = 5
y = int(x)
for i in range(1,y+2):
    print((y+1-i)*" ",end="")
    d=1
    for j in range(1,i+1):
        print(d,end=" ")
        d = int(d*(i-j)/j)
    print("")
        


# In[ ]:


# Pattern 30:
x = 5
y = int(x)
for i in range(0,y):
    for j in range(1,i+1):
        print(j,end=" ")
    for j in range(i+1,0,-1):
        print(j,end=" ")
    print(" ")       


# In[ ]:


# Pattern 31:
x = 5
y = int(x)
for i in range(1,y+1):
    print(" "*(i-1),end =" ")
    for j in range(1,y+2-i):
        print(j,end=" ")
    print("")
    


# In[ ]:


# Pattern 32:
x = 5
y = int(x)
for i in range(1,y):
    print(" "*(y-i),end=" ")
    for j in range(1,i+1):
        print(j,end=" ")
    print("")
for k in range(1,y+1):
    print(" "*(k-1),end =" ")
    for l in range(1,y+2-k):
        print(l,end=" ")
    print("")

