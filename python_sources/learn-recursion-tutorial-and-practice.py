#!/usr/bin/env python
# coding: utf-8

# #Recursion
# This Kernel will attempt to guide you through learning the basics of recursion. You will learn:
# 1. Some intuition on approaching recursive algorithms
# 2. The basic structure of a recursive solution
# 3. Recursive primitives like traversals and depth first search
# 4. Additional recursive building blocks for graph search and dynamic programming
# 5. How to modify and combine recursive primitives to solve most problems
# 6. How to approach more complex recursive problems on your own
# 
# **Prerequisites**
# 1. You should already have a basic understanding of python, including loops, classes and functions.
# 2. You should know about theses data structures and common variants: trees, graphs, lists, linked lists, dictionaries

# ## Section 1 - Thinking Recursively
# Recursion is powerful because we can solve hard problems by thinking small. When done correctly, thinking recursively will make you feel like a wizard. You follow a few steps and a solution to a complex problem will magically emerge. That is how I feel when I approach and solve a challenging recursion problem. You will too, with a bit of practice.
# 
# ### What is recursion?
# Fundamentally, recursion is the act of a function calling itself. There are many reasons to use recursion, but in my experience, these are the main classes of recursion you're likely to encounter in a job or interview:
# 1. Graphs
# 2. Trees
# 3. Loops
# 4. Branching algorithms* (basically trees)
# 
# *TODO: I need to expand on this section. Please upvote the kernel if you find it useful**

# ## Section 2 - The Basic Structure of a Recursive Solution
# A recursive function generally has three parts:
# 1. **A base case** - A statement that prevents the recursion from running indefinitely
# 2. **An operation** - Do the main thing you're writing your solution to do
# 3. **A recursive call** - Now recurse
# 
# In an interative world these are roughly equivalent to:
# 1. A condition
# 2. An operation (loop body)
# 3. An increment
# 
# The following two functions are analagous. See how #1, #2 and #3 align.

# In[ ]:


def iterative_counter(initial_value):
    num = initial_value
    # 1. The condition
    while (num < 5):
        
        # 2 The operation
        print(num, end=" ")
        
        # 3. The increment
        num += 1
    
def recursive_counter(num):
    # 1. The base case
    if (num >= 5):
        return
    
    # 2. The operation
    print(num, end=" ")
    
    # 3. The recursive call
    recursive_counter(num + 1)

print("Iteration: ")
iterative_counter(0)

print("\nRecursion: ")
recursive_counter(0)


# ** Exercise 1**
# 
# In the recursive example above, the base case in an inversion of the while conditional, because it is a **stopping** conditional whereas the while is a **continuing** conditional. *Refactor* the recursive method to use the same conditional as the while loop. Remember that if you invert the conditional, you'll need to put the recursive call and operation both inside the if block.

# ## Section 3 - Recursive Primitives
# This section aims to cover a few basic recursion algorithms that will form the basis of most of the recursive solutions you ever write
# 
# ### Tree Traversal
# One of the most foundational algorithms is the *traversal*. In this case, we'll use a binary tree as the thing to traverse:

# In[ ]:


class BinaryTreeNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
        
binary_tree = BinaryTreeNode(1, 
                             BinaryTreeNode(2, BinaryTreeNode(3), BinaryTreeNode(4)), 
                             BinaryTreeNode(5))


# In[ ]:


def binary_tree_preorder_traversal(node):
    # 1. The base case
    if node is None:
        return
    
    # 2. The operation
    print(node.value, end=" ")
    
    #3. The recursive call(s)
    binary_tree_preorder_traversal(node.left)
    binary_tree_preorder_traversal(node.right)

binary_tree_preorder_traversal(binary_tree)


# The traversal above follows the structure as any other recursion. It has a base case, an operation and a recursive call. Here, we have two recursive calls, allowing our algorithm to print both children nodes.
# 
# Recursive methods generallly have three parts, but the parts don't have to be in order. For example, we could print our nodes in a different order by splitting up the recursive calls!

# In[ ]:


def binary_tree_inorder_traversal(node):
    # 1. The base case
    if node is None:
        return
    
    # 3. The recursive call part 1
    binary_tree_inorder_traversal(node.left)
    
    # 2. The operation
    print(node.value, end=" ")
    
    # 3. The recursive call part 2
    binary_tree_inorder_traversal(node.right)

binary_tree_inorder_traversal(binary_tree)


# First, we defined a *preorder* traversal that would print a node, then print the left subtree, then print the right subtree. Next, we defined an *inorder* traversal which prints the left subtree first, then the node, then the right subtree. An inorder traversal is very important for iterating over a *binary search tree* which is a tree where the left subtree is smaller than the node and the right subtree is all greater than the node. 
# 
# **Exercise 2**
# 
# Implement a *postorder* traversal. In a post-order traversal, the subtrees will be printed before the node.
# 
# Expected output: 3 4 2 5 1

# In[ ]:


def binary_tree_postorder_traversal(node):
    # Implement here
    return
    
binary_tree_inorder_traversal(binary_tree)


# When thinking about binary trees, you will likely read about these three traversals:
# 1. Preorder *Node, Left, Right*
# 2. Inorder *Left, Node, Right*
# 3. Postorder *Left, Right, Node*
# 
# Generally, for a tree problem, always visit the left before the right unless there
# is a good reason to do otherwise. 
# 
# A traversal doesn't have to only print, and in some cases you might be asked to modify the tree.
# 
# **Exercise 3**
# 
# Implement a recursive method that will add 1 to every node.
# 
# *Hint: Follow the pattern of any of the three traversals above, but replace the operation to update node.value in place*
# 
# Expected Output: 2 3 4 5 6
# 

# In[ ]:


def add_one_binary_tree(node):
    # Implement here
    return None
    
    
# I will make a copy of the binary tree for your method to mutate
import copy
new_binary_tree = copy.deepcopy(binary_tree)
add_one_binary_tree(new_binary_tree)

# Print an answer
binary_tree_preorder_traversal(new_binary_tree)


# ### Depth First Search (DFS)
# The next foundational recursive algorithm is the depth first search. This section will introduce the idea of returning values back up the recursive stack. A depth first search looks like this:

# In[ ]:


def depth_first_search(node, search_value):
    # 1. The base case
    if node is None:
        return None;
    
    # 2. The operation
    if node.value == search_value:
        return node
    
    # 3. The recursive call(s)
    left = depth_first_search(node.left, search_value)
    if left is not None:
        return left
    
    right = depth_first_search(node.right, search_value)
    if right is not None:
        return right


# A DFS algorithm returns a node who matches some condition (in this case, values must match). However, the node you're looking for probably isn't the root node. In that case, you have to return the values back up the stack. That's what is happening in the recursive calls. 
# 
# To help practice returning things up the stack, complete the following exercise on returning values.
# 
# **Exercise 4**
# 
# Finish implementing the following recursive function which returns the most-left value in a tree.
# 
# Expected Output: 3

# In[ ]:


def leftmost_value(node):
    # 1. Base Case

    # 2. Operation - If I can't go left anymore, I am the leftmost.
    if node.left is None:
        return node.value;
    
    # 3. Recursive Case
    

print(leftmost_value(binary_tree))


# ### Other tree problems
# There is a broad class of tree problems that require you to process a tree in some way.
# 
# This class of problems is most commonly solved with a post-order approach.
# 
# In the following problem, we are asked to count the number of nodes in a binary tree. It can help to think locally. Think like node function, and the solution emerges:
# 
# "I only know one node"  
# "I have a function that counts nodes"  
# "I will use that function to find out how many nodes are on my left"  
# "I will use that function to find out how many nodes are on my right"  
# "My subtree has that many nodes, plus one, for me"  
# 

# In[ ]:


def count_nodes(node):
    # 1. Base case
    if node is None:
        return 0
    
    # 3. Recursive calls
    left_node_count = count_nodes(node.left)
    right_node_count = count_nodes(node.right)
    
    #2. The operation (Post-Order)
    subtree_node_count = left_node_count + right_node_count + 1
    return subtree_node_count

count_nodes(binary_tree)


# The following exercises will help you practice:
# 
# **Exercise 5**
# 
# Finish implementing a recursive function that returns the sum of all nodes in a binary tree:
# 
# Expected: 15

# In[ ]:


def sum_nodes(node):
    # 1. Base case
    if node is None:
        return 0

    # Implement here
    

print(sum_nodes(binary_tree))


# Choosing a good base case can be tricky!
# 
# **Exercise 7**
# 
# The following function does not work correctly because a bad base case was chosen.
# 
# *Hint: If you decide to return None from the base case, make sure to handle the possibility that left_max or right_max could be none!*
# 
# Expected: -2

# In[ ]:


def find_max_value(node):
    # 1. Base case
    if node is None:
        return 0
    
    # 3 Recursive calls
    left_max = find_max_value(node.left)
    right_max = find_max_value(node.right)
    
    # 2. Operation (post-order)
    my_max = node.value
    if left_max > my_max:
        my_max = left_max
    if right_max > my_max:
        my_max = right_max
        
    return my_max
    
print("For a positive tree, I found:")
print(find_max_value(binary_tree),)

negative_binary_tree = BinaryTreeNode(-2, BinaryTreeNode(-3), BinaryTreeNode(-4))
print("For a negative tree, I found:")
print(str(find_max_value(negative_binary_tree)) + " (expected: -2)")

