#!/usr/bin/env python
# coding: utf-8

# In[72]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[73]:


#Algo1
class LifeGrid:
    
    #initial constant DEAD CELL and LIVE CELL
    DEAD_CELL=0
    LIVE_CELL=1
    
    def __init__(self,rows,cols):
        #initial nrow and ncol
        self._rows=rows
        self._cols=cols
        #initial grid size nrow*ncol as None
        self._grid=[[0]*cols for i in range(rows)]
        #set all element in grid as DEAD CELL
        self.configure(list())
        
        
    #method that return nrow and ncol
    def numRows(self):
        return self._rows
    def numCols(self):
        return self._cols
    
    #configure status of cell from live cell in coordList
    def configure(self,coordList):
        #set status of all cell to dead
        for i in range(self.numRows()):
            for j in range(self.numCols()):
                self.clearCell(i,j) 
        #set status of specific cell as LIVE CELL
        for coord in coordList:
            self.setCell(coord[0],coord[1])
        
    #method that check whether it's live cell
    def isLiveCell(self,row,col):
        return self[row][col]==self.LIVE_CELL
    
    #method that clear that cell to DEAD CELL
    def clearCell(self,row,col):
        self[row][col]=self.DEAD_CELL
        
    #method that set that cell to LIVE CELL
    def setCell(self,row,col):
        self[row][col]=self.LIVE_CELL
        
    #method that check live cell around it
    def numLiveNeighbors(self,row,col):
        #this line is same as
        #   ans=0
        #   for i in range(max(0,row-1),min(row+1,self.numRows()-1)+1):
        #       for j in range(max(0,col-1),min(col+1,self.numCols()-1)+1):
        #           if self.isLiveCell(i,j):
        #               ans+=1
        #why must use range(max(0,row-1),min(row+1,self.numRows()-1)+1)?
        #   ans -> in normal state will check as
        #           i-1 i i+1
        #           ?   ?   ?
        #           ?   *   ?
        #           ?   ?   ?
        #           if row want to find is 0
        #           -1  0 +1
        #            ?  ?  ?
        #            ?  *  ?
        #            ?  ?  ?
        #           so should start from 0
        #           if row want to find is nrow-1
        #           nrow-2  nrow-1  norw
        #           ?       ?       ?
        #           ?       *       ?
        #           ?       ?       ?
        #           so shouhld end at nrow-1
        #       col is same as row
        ans = sum([1 for j in range(max(0,col-1),min(col+1,self.numCols()-1)+1)                 for i in range(max(0,row-1),min(row+1,self.numRows()-1)+1) if                 self.isLiveCell(i,j)])
        #return ans-1 if itself is LIVE CELL else return ans
        return (ans-1 if self.isLiveCell(row,col) else ans)
        
    #special method that can access row directly, dont need self._grid anymore
    def __getitem__(self,row):
        return self._grid[row]

#Algo2
class LifeGrid2:
    
    def __init__(self,rows,cols):
        #initial nrow and ncol
        self._rows=rows
        self._cols=cols
        #initial dictionary that collect only live cell
        self._liveDict=dict()
        #initial grid rows*cols
        self._grid=np.zeros([rows,cols])
        #set all element in grid as DEAD CELL
        self.configure(list())
        
    #method that return nrow and ncol
    def numRows(self):
        return self._rows
    def numCols(self):
        return self._cols
    
    #configure status of cell from live cell in coordList
    def configure(self,coordList):
        #set status of all cell to dead
        self._liveDict.clear()
        #set all elements in grid as 0
        self._grid=np.zeros([self.numRows(),self.numCols()])
        #set status of specific cell as LIVE CELL
        for coord in coordList:
            self.setCell(coord[0],coord[1])
            self[coord[0]][coord[1]]=1
        
    #method that check whether it's live cell
    def isLiveCell(self,row,col):
        return self[row][col]==1
    
    #method that clear that cell to DEAD CELL
    def clearCell(self,row,col):
        #set value of that cell as 0
        self._grid[row][col]=0
        #these lines below has meaning as
        #   if (row,col) in self._liveDict.keys():
        #      del self._liveDict[(row,col)]
        #but I love try-except 55555
        try:
            del self._liveDict[(row,col)]
        except:
            return ;
        
    #method that set that cell to LIVE CELL
    def setCell(self,row,col):
        self._liveDict[(row,col)]=1
        self._grid[row][col]=1
        
    #method that check live cell around it
    def numLiveNeighbors(self,row,col):
        #this line is same as
        #   ans=0
        #   for i in range(max(0,row-1),min(row+1,self.numRows()-1)+1):
        #       for j in range(max(0,col-1),min(col+1,self.numCols()-1)+1):
        #           if self.isLiveCell(i,j):
        #               ans+=1
        #why must use range(max(0,row-1),min(row+1,self.numRows()-1)+1)?
        #   ans -> in normal state will check as
        #           i-1 i i+1
        #           ?   ?   ?
        #           ?   *   ?
        #           ?   ?   ?
        #           if row want to find is 0
        #           -1  0 +1
        #            ?  ?  ?
        #            ?  *  ?
        #            ?  ?  ?
        #           so should start from 0
        #           if row want to find is nrow-1
        #           nrow-2  nrow-1  norw
        #           ?       ?       ?
        #           ?       *       ?
        #           ?       ?       ?
        #           so shouhld end at nrow-1
        #       col is same as row
        ans = sum([1 for j in range(max(0,col-1),min(col+1,self.numCols()-1)+1)                 for i in range(max(0,row-1),min(row+1,self.numRows()-1)+1) if                 self.isLiveCell(i,j)])
        #return ans-1 if itself is LIVE CELL else return ans
        return (ans-1 if self.isLiveCell(row,col) else ans)
    
    #function that abbreviate self._grid[row][col] to self[row][col]
    def __getitem__(self,row):
        return self._grid[row]

#Algo3

import ctypes

class Array:
    def __init__(self,size,show_status=False):
        assert size>0,"Array size must be > 0"
        self._size=size
        PyArrayType=ctypes.py_object*size
        self._elements=PyArrayType()
        self.clear(None)
        if show_status:
            print("\n1D Array Initial")
    def __len__(self):
        return self._size
    def __getitem__(self,index):
        assert index>=0 and index<len(self),"INdex Out Of Range"
        return self._elements[index]
    def __setitem__(self,index,value):
        assert index>=0 and index<len(self),"Index Out Of Range"
        self._elements[index]=value
    def clear(self,value):
        for i in range(len(self)):
            self._elements[i]=value
    def __iter__(self):
        return _ArrayIterator(self._elements)
    
class _ArrayIterator:
    def __init__(self,array):
        self._arrayRef=array
        self._curNdx=0
    def __iter__(self):
        return self
    def __next__(self):
        if self._curNdx<len(self._arrayRef):
            entry=self._arrayRef[self._curNdx]
            self._curNdx+=1
            return entry
        else:
            raise StopIteration
            
class Array2D:
    def __init__(self,nrow,ncol,show_status=False):
        self._row=Array(nrow)
        for i in range(nrow):
            self._row[i]=Array(ncol)
        if show_status:
            print("\n2D Array Initial")
    def numRows(self):
        return len(self._row)
    def numCols(self):
        return len(self._row[0])
    def clear(self,value):
        for i in range(self.numRows()):
            self._row[i].clear(value)
    def __getitem__(self,ndxTuple):
        assert len(ndxTuple)==2,"Invalid number of array subscripts"
        row,col=ndxTuple[0],ndxTuple[1]
        assert row>=0 and row<self.numRows() and col>=0 and col<self.numCols(),        "Array subscript out of range"
        the1da=self._row[row]
        return the1da[col]
    def __setitem__(self,ndxTuple,value):
        assert len(ndxTuple)==2,"Invalid number of array subscripts"
        row,col=ndxTuple[0],ndxTuple[1]
        assert row>=0 and row<self.numRows() and col>=0 and col<self.numCols(),        "Array subscript out of range"
        the1da=self._row[row]
        the1da[col]=value

class LifeGrid3:
    
    def __init__(self,rows,cols):
        #initial nrow and ncol
        self._rows=rows
        self._cols=cols
        #initial dictionary that collect only live cell
        self._liveDict=dict()
        #initial grid rows*cols
        self._grid=Array2D(rows,cols)
        #set all element in grid as DEAD CELL
        self.configure(list())
        
    #method that return nrow and ncol
    def numRows(self):
        return self._rows
    def numCols(self):
        return self._cols
    
    #configure status of cell from live cell in coordList
    def configure(self,coordList):
        #set status of all cell to dead
        self._liveDict.clear()
        #set all elements in grid as 0
        for i in range(self.numRows()):
            for j in range(self.numCols()):
                self.clearCell(i,j)
        #set status of specific cell as LIVE CELL
        for coord in coordList:
            self.setCell(coord[0],coord[1])
        
    #method that check whether it's live cell
    def isLiveCell(self,row,col):
        return self._grid[row,col]==1
    
    #method that clear that cell to DEAD CELL
    def clearCell(self,row,col):
        #set value of that cell as 0
        self._grid[row,col]=0
        #these lines below has meaning as
        #   if (row,col) in self._liveDict.keys():
        #      del self._liveDict[(row,col)]
        #but I love try-except 55555
        try:
            del self._liveDict[(row,col)]
        except:
            return ;
        
    #method that set that cell to LIVE CELL
    def setCell(self,row,col):
        self._liveDict[(row,col)]=1
        self._grid[row,col]=1
        
    #method that check live cell around it
    def numLiveNeighbors(self,row,col):
        #this line is same as
        #   ans=0
        #   for i in range(max(0,row-1),min(row+1,self.numRows()-1)+1):
        #       for j in range(max(0,col-1),min(col+1,self.numCols()-1)+1):
        #           if self.isLiveCell(i,j):
        #               ans+=1
        #why must use range(max(0,row-1),min(row+1,self.numRows()-1)+1)?
        #   ans -> in normal state will check as
        #           i-1 i i+1
        #           ?   ?   ?
        #           ?   *   ?
        #           ?   ?   ?
        #           if row want to find is 0
        #           -1  0 +1
        #            ?  ?  ?
        #            ?  *  ?
        #            ?  ?  ?
        #           so should start from 0
        #           if row want to find is nrow-1
        #           nrow-2  nrow-1  norw
        #           ?       ?       ?
        #           ?       *       ?
        #           ?       ?       ?
        #           so shouhld end at nrow-1
        #       col is same as row
        ans = sum([1 for j in range(max(0,col-1),min(col+1,self.numCols()-1)+1)                 for i in range(max(0,row-1),min(row+1,self.numRows()-1)+1) if                 self.isLiveCell(i,j)])
        #return ans-1 if itself is LIVE CELL else return ans
        return (ans-1 if self.isLiveCell(row,col) else ans)


# In[74]:


print("start algorithm1")
def evolve(grid):
    #initialize liveCells as empty list
    liveCells=list()
    #access every cells in grid
    for i in range(grid.numRows()):
        for j in range(grid.numCols()):
            #count neighbors that is LIVE CELL around (i,j)
            neighbors=grid.numLiveNeighbors(i,j)
            #if neighbors is 2 and (i,j) has live or neighbors is 3
            if (neighbors==2 and grid.isLiveCell(i,j)) or neighbors==3:
                #this cell will live in next turn
                liveCells.append((i,j))
    #configure grid with new live Cell
    grid.configure(liveCells)
    
from time import time
import seaborn as sns;sns.set()
import numpy as np
import matplotlib.pyplot as plt

#initialize generation of game ( can change to input later )
NUM_GEN=10

n=[10,100,1000]
ans1=[]
for i in n:
    print("FOR n =",i)
    INIT_CONFIG=[(np.random.randint(0,i),np.random.randint(0,i)) for j in range(int(i*i*0.7))]
    grid=LifeGrid(i,i)
    grid.configure(INIT_CONFIG)
    start=time()
    for i in range(NUM_GEN):
        #evolve grid
        evolve(grid)
        #draw grid
        #draw(grid)
    endd=time()
    ans1.append(endd-start)
    print(endd-start,"seconds")


# In[75]:


sns.lineplot(x=n,y=ans1,label="Algo1")
print("Start Algo2")
#initialize generation of game ( can change to input later )
NUM_GEN=10

#function that evolve grid for one turn
def evolve(grid):
    #initialize liveCells as empty set
    liveCells=set()
    #initial unknown cell as empty set
    unknownCells=set()
    #iteration in every item in liveDict
    for key,_ in grid._liveDict.items():
        #add all dead cell around this live cell to unknownCells
        unknownCells.update([(i,j) for j in range(max(0,key[1]-1),min(key[1]+1                              ,grid.numCols()-1)+1) for i in range(max(0,key[0]                                           -1),min(key[0]+1,grid.numRows()-1)+1) if not grid.isLiveCell(i,j)])
        #find number of neighbors around this cell
        neighbors=grid.numLiveNeighbors(key[0],key[1])
        #check neighbors from condition , if true then add to liveCells
        if (neighbors==2 and grid.isLiveCell(key[0],key[1])) or neighbors==3:
            liveCells.add(key)
    #iteration in every unknownCells
    for i,j in unknownCells:
        #find neighbors of this cell
        neighbors=grid.numLiveNeighbors(i,j)
        #check condition with neighbors , if true then add to liveCells
        if (neighbors==2 and grid.isLiveCell(i,j)) or neighbors==3:
            liveCells.add((i,j))
    #configure grid with liveCells
    grid.configure(list(liveCells))
    
for p in [10,20,30,40,50,60,70,80,90]:
    ans=[]
    for j in [10,100,1000]:
        print("For n =",j,",",p,"%")
        INIT_CONFIG=[(np.random.randint(0,j),np.random.randint(0,j)) for k in range(int(j*j*p/100))]
        #initialize grid with HEIGHT AND WIDTH
        grid=LifeGrid2(j,j)
        #configure grid with initial live cell
        grid.configure(INIT_CONFIG)
        start=time()
        #describe how to do in each gen
        for i in range(NUM_GEN):
            #evolve grid
            evolve(grid)
            #draw grid
            #draw(grid)
        endd=time()
        ans.append(endd-start)
        print(endd-start,"seconds")
    sns.lineplot(x=[10,100,1000],y=ans,label="Algo2_"+str(p)+"%")
plt.title("Time Comparison")
plt.xlabel("number of height and width")
plt.ylabel("time(sec)")
plt.show()


# In[ ]:




