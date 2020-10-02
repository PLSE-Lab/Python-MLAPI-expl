# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

def validList(x, y, grid):
    """
    Returns a list of possible numbers for given position x, y and numpy array
    """
	outlist = []
	for n in range(1, 10):
		if n in grid[y,:]:
			continue
		if n in grid[:,x]:
			continue
		if n in grid[3*int(y/3):(3*int(y/3))+3, 3*int(x/3):(3*int(x/3))+3]:
			continue
		outlist.append(n)
	return outlist

def toArr(ss):
    """
    Just a function to transform a string into a numpy array
    """
	c = []
	for ind, s in enumerate(ss):
		if len(c) <= ind/9:
			c.append([])
		c[int(ind/9)].append(int(s))
	return np.array(c)

def checkSolved(grid):
    """
    Checks if sudoku is solved (doesn't have to be correct)
    """
	return not 0 in grid

def checkCorrect(grid):
    """
    Checks if sudoku is solved correctly
    """
	perfect = np.array([1,2,3,4,5,6,7,8,9])
	for x in range(0, 9):
		l = sorted(grid[:,x])
		if not np.array_equal(l, perfect):
			return False
	for y in range(0, 9):
		l = sorted(grid[y,:])
		if not np.array_equal(l, perfect):
			return False
	for sq in range(0,9):
		l = sorted(grid[(sq%3)*3 : ((sq%3)*3)+3, (3*int(sq/3)):(3*int(sq/3))+3].flatten())
		if not np.array_equal(l, perfect):
			return False
	return True

def easySolve(instring):
    """
    An algorithm to solve easy sudokus. It just checks if any of the squares has only one possible number and repeats.
    I could probably use parts of it to make my final solver faster.
    """
	grid = toArr(instring)
	unsolved = True
	while unsolved:
		anychange = False
		for x in range(0, 9):
			for y in range (0,9):
				if grid[y][x] == 0:
					l = validList(x, y, grid)
					if len(l) == 1:
						grid[y][x] = l[0]
						anychange = True
						unsolved = not checkSolved(grid)
				if not unsolved:
					break
			if not unsolved:
				break
		if not anychange:
			break
	return ''.join(map(str, grid.flatten().tolist()))

def solve(instring):
    grid = toArr(instring)
    """
    A recursive algorithm to check all the available numbers. I'll optimize it one day, I swear!
    """
    if checkSolved(grid):
        return grid
    x,y = -1, -1
    for xx in range(0, len(grid)):
        for yy in range(0, len(grid[0])):
            if grid[yy][xx] == 0:
                x ,y = xx, yy
                break
        if x > -1:
            break
    if x == -1 or y == -1:
        return grid
    l = validList(x, y, grid)
    for n in l:
        grid[y][x] = n
        z = solver(grid)
        if checkSolved(z):
            return z
    grid[y][x] = 0
    return ''.join(map(str, grid.flatten().tolist()))

import csv

"""
Here I call the function. I should probably parallelize the calls.
"""
with open('../input/sudoku.csv', newline='') as f:
    x = csv.reader(f)
    results = []
    n = 0
    for r in x:
        if r[0] == 'quizzes':
            continue
        q, a = r[0], r[1]
        assert solve(q) == a, "FAIL!"