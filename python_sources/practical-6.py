#!/usr/bin/env python
# coding: utf-8

# # Q.1
# 
# - a) /user/rt/courses
# - b) Internal nodes are any nodes that has a chil node
# - c) cs016/ has 9 decendants
# - d) cs016/ has 1 ancestor, the /user/rt/courses node.
# - e) the programs/ and grades/ nodes
# - f) papers/ , bylow, sellhigh and demos/ , market
# - g) the depth of papers is 3. 
# - h) the height of the tree is 5. 

# # Q.2
# 
# - a) Inorder(Left,Root,Right): 4,2,5,1,8,6,9,3,7,10
# - b) Preorder (Root,Left,Right): 1,2,4,5,3,6,8,9,7,10
# - c) Postorder (Left,Right,Root): 4,5,2,8,9,6,10,7,3,1

# In[2]:


get_ipython().system('pip install binarytree')
#!pip install anytree


from binarytree import Node

root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.left = Node(4)
root.left.right = Node(5)
root.right.left = Node(6)
root.right.right = Node(7)
root.right.left.left = Node(8)
root.right.left.right = Node(9)
root.right.right.right = Node(10)
                             
print("Root height: ", root.height)
print("Inorder: ", root.inorder)
print("Preorder: ", root.preorder)
print("Postorder: ", root.postorder)
print(root)


