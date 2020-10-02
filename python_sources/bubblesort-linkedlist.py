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
###############################---- Bubble Sort Singly Linked List -----##########################################

# Node consists of a DataValue and NextValue

class Node:

    def __init__(self, dataval = None):
        self.dataval = dataval
        self.nextval = None

# Linked List consists of HeadValue
# HeadValue should always be nodes


class LinkedList:

    def __init__(self):
        self.headval = None

# printlist() is to see the data inside the LinkedList
    def printlist(self):
        val = self.headval
        while val:
            print(val.dataval)
            val = val.nextval

# Move HeadValue of LinkedList to NextValue of newNode and create a reference to newNode on HeadValue of LinkedList
    def insert_at_beginning(self, newval):
        newNode = Node(newval)
        newNode.nextval = self.headval
        self.headval = newNode

# insert at end
    def insert_at_end(self, newval):
        newNode = Node(newval)
        # check if headvalue is None
        if self.headval is None:
            self.headval = newNode
            return
        # at least one or more nodes
        lastNode = self.headval
        while lastNode.nextval:
            lastNode = lastNode.nextval
        lastNode.nextval = newNode

# insert in middle
    def insert_after_node(self, node, nextval):
        # node is before newNode
        # check if node is None
        if node is None:
            print("Node is empty, cannot add after empty node")
            return
        newNode = Node(nextval)
        newNode.nextval = node.nextval
        node.nextval = newNode

# remove node
    def remove_node(self, node_to_remove):
        if node_to_remove is None:
            print("Node is empty, cannot remove empty node")
            return
        node = self.headval
        while node.nextval:
            if node.nextval == node_to_remove:
                node.nextval = node_to_remove.nextval
                ## cleanup
                node_to_remove.nextval = None
                node_to_remove.dataval = None
                ##
                return
            # move to nextvalue else endless loop
            node = node.nextval

    def bubble_sort(self):
        last = None
        while self.headval != last:
            x = self.headval
            while x.nextval != last:
                y = x.nextval
                if x.dataval > y.dataval:
                    x.dataval, y.dataval = y.dataval, x.dataval
                x = x.nextval
            last = x

# instantiation of LinkedList
myList = LinkedList()
"""
# instantiation of Node in LinkedList
myList.headval = Node("A")

# more Nodes
nodeB = Node("B")
nodeC = Node("C")

# Linking Nodes to linkedList
myList.headval.nextval = nodeB

# directly accessing Node B and linking to Node C
nodeB.nextval = nodeC

# Since Node C is the last element, nextval of Node C is None

# insert element at beginning
myList.insert_at_beginning("NewNode")

# insert element at end
myList.insert_at_end("NewEndNode")

# insert element after node
myList.insert_after_node(nodeB, "NewInsertedNode")

# remove a node
myList.remove_node(nodeB)
"""
# insert elements
myList.insert_at_beginning(70)
myList.insert_at_beginning(34)
myList.insert_at_beginning(3)
myList.insert_at_beginning(87)
myList.insert_at_beginning(84)
myList.insert_at_beginning(60)
myList.insert_at_beginning(19)
myList.insert_at_beginning(10)
myList.insert_at_beginning(4)
myList.insert_at_beginning(2)
myList.insert_at_beginning(86)
myList.insert_at_beginning(2)
myList.insert_at_beginning(88)
myList.insert_at_beginning(81)
myList.insert_at_beginning(45)
myList.insert_at_beginning(96)
myList.insert_at_beginning(16)
myList.insert_at_beginning(15)
myList.insert_at_beginning(20)
myList.insert_at_beginning(35)



# print LinkedList
myList.printlist()

myList.bubble_sort()

print("SORTED")
myList.printlist()