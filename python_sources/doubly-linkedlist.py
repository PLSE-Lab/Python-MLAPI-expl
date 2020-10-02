# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

##########################################Python Doubly-LinkedList ###############################################
# Node consists of previousValue, dataValue and nextValue


class Node:
    def __init__(self, data=None):
        self.prevval = None
        self.dataval = data
        self.nextval = None


class DoublyLinkedList:
    def __init__(self):
        self.headval = None

    def printlist(self):
        val = self.headval
        while val:
            print(val.dataval)
            val = val.nextval

    def insert_at_beginning(self, newval):
        newNode = Node(newval)

        # set nextValue to old headvalue
        newNode.nextval = self.headval

        #set prevValue to None
        newNode.prevval = None
        # check for empty case
        if self.headval is not None:
            self.headval.prev = newNode
        # update headValue
        self.headval = newNode

    def insert_at_end(self, newval):
        newNode = Node(newval)
        # since its end of list, nextValue = None
        newNode.nextval = None
        # check for empty case
        if self.headval is None:
            newNode.prevval = None
            self.headval = newNode
            return

        val = self.headval
        #end of list
        while val.nextval:
            val = val.nextval

        # set nextval to newNode
        val.nextval = newNode
        newNode.prevval = val


    def insert_after(self, node, newval):
        # check empty list case
        if self.headval is None:
            print("List is empty")
            return

        newNode = Node(newval)

        newNode.nextval = node.nextval
        node.nextval = newNode
        newNode.prevval = node

        if newNode.nextval != None:
            newNode.nextval.prevval = newNode
        else:
            self.headval = newNode

"""
    def get_size_list(self):
        count = 0
        val = self.headval
        while val:
            count += 1
            val = val.nextval
        return count
"""

doubleList = DoublyLinkedList()
node = Node("First Node")
doubleList.headval = node

# insert at beginning
doubleList.insert_at_beginning("E")
doubleList.insert_at_beginning("D")
doubleList.insert_at_beginning("C")
doubleList.insert_at_beginning("B")
doubleList.insert_at_beginning("A")

# insert at end
doubleList.insert_at_end("NewEndNode")

# insert after
doubleList.insert_after(node, "NewAfterNode")

# print the list
doubleList.printlist()