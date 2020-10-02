#!/usr/bin/env python
# coding: utf-8

# ### 0. Creating a set
# 
# We can easily create a set just by inputing directly the values in the curly braces: `{}`

# In[ ]:


set_A = {1, 2, 3, "hello", "cut"}
type(set_A)


# using `set(list of something)`

# In[ ]:


set_B = set([1, 2, 3, "hello", "cut"])
type(set_B)


# or `set(numpy_array of something)`

# In[ ]:


import numpy as np
set_C = set(np.arange(0, 3, 1))
type(set_C)


# ### 1. complement and (regular) difference.
# 
# Let $S$ be a universal set, and A, B be its subsets, we have
# 
# $$ A \setminus B = \lbrace x \in A : x \not \in B \rbrace $$
# 
# Moreover, the set $A^C$ defined by
# 
# $$ A^C = S \setminus A = \lbrace x \in S : x \not \in A \rbrace $$
# 
# is called the complement of $A$ in $S$.

# In[ ]:


S = set(["cat1", "dog1", "cat2", "dog2", "cat3", "cat4", 2j, False, 1, 1j+2, 2.5, 1+2j])
A = set(["cat1", "dog1", "cat2", "dog2", 1, 2.5, False, 1+2j])
B = set(["cat1", "dog1", "cat2", "dog2", "cat3", "cat4", 2j])


# We can use both of 
# 
#                         S.difference(A) 
# and
# 
#                         S - A 
# to simplify the expression $S \setminus A$

# In[ ]:


print(S.difference(A))
print(S - A)


# In[ ]:


print(S - A)
print(S - B)


# But, noting that the `set_difference` is not **`symmetric`**, that is
# $$ A \setminus B \neq B \setminus A $$
# iff $A \neq B$

# In[ ]:


print(B - A)
print(A - B)


# Obviousky, $A \setminus A = \emptyset$

# In[ ]:


A - A


# ### 2. Intersection.
# 
# 
# We have  $$ A \cap B = B \cap A = \lbrace x \in A \text{ and } x \in B \rbrace = A \setminus ( A \setminus B ) = B \setminus ( B \setminus A ) $$
# 

# In[ ]:


print(A - (A - B))
print(B - (B - A))
print(B.intersection(A))
print(A.intersection(B))


# Moreover, we can call `A & B` to simplify the expression $A \cap B$

# In[ ]:


A & B


# ### 3. Union.
# 
# We have  $$ A \cup B = B \cup A = \lbrace x \in A \text{ or } x \in B \rbrace $$

# In[ ]:


print(A.union(B))
print(B.union(A))


# Likewise, we can call `A | B` to simplify the expression $A \cup B$

# In[ ]:


print(A | B)


# ### 4. Symmetric difference.
# 
# The `symmetric difference` is equivalent to the union of both relative complements, that is:
# 
# $${\displaystyle A\,\triangle \,B=\left(A\setminus B\right)\cup \left(B\setminus A\right) = \displaystyle B \triangle \displaystyle A,}$$

# In[ ]:


print(A.symmetric_difference(B))
print(B.symmetric_difference(A))
print('Verify: \n')
(A - B).union(B - A)


# **Properties.**
# 
# 1. The symmetric difference is commutative and associative:
# 
# $${\displaystyle {\begin{aligned} A\,\triangle \,B &= B\,\triangle \,A \,\\(A\,\triangle \,B)\,\triangle \,C &= A\,\triangle \,(B\,\triangle \,C).\end{aligned}}}$$
# 
# Moreover,
# 
# $$ A\,\triangle \,B = A^C \, \triangle B^C.$$
# 
# 2. $A \cup B$ can be expressed by the disjoint-union of $A \cap B$ and $A\,\triangle \,B$, So the symmetric difference can also be expressed as the union of the two sets, minus their intersection:
# 
# $$ {\displaystyle A\,\triangle \,B=(A\cup B)\setminus (A\cap B),}$$
# 
# 3. The empty set is neutral, and every set is its own inverse:
# 
# $${\displaystyle {\begin{aligned}A\,\triangle \,\emptyset &=A, \end{aligned}}} \\ {\displaystyle {\begin{aligned}A\,\triangle A\ &= \emptyset .\end{aligned}}}$$
# 
# 4. Distributes over symmetric difference
# 
# $$ {\displaystyle (A\,\triangle \,B)\,\triangle \,(B\,\triangle \,C)=A\,\triangle \,C.} $$
# 
# $$ A\cap (B\,\triangle \,C)=(A\cap B)\,\triangle \,(A\cap C),$$
# 
# 5. $\left(\bigcup _{\alpha \in {\mathcal {I}}}A_{\alpha }\right)\triangle \left(\bigcup _{\alpha \in {\mathcal {I}}}B_{\alpha }\right)\subseteq \bigcup _{\alpha \in {\mathcal {I}}}\left(A_{\alpha }\triangle B_{\alpha }\right),$ where $\displaystyle {\mathcal {I}}$ is an arbitrary non-empty index set.

# In[ ]:


print('Illustration example')
##############################################
A = set([1 , 2, "1"])
B = set([2, "2", 1])
C = set(["3", 1, 2])
S = set([1,2,3, "1", "2", "3"])
E = set([])
print('Let S = %s be the sample_space and \n \t\t\t A = %s, B = %s, C = %s be its subsets'%(S , A, B, C))
##############################################
print('properties 1: \n\t  i) A sym_diff B = %s = %s = B sym_diff A '%(
    A.symmetric_difference(B), B.symmetric_difference(A)))
print('\t ii) (A sym_diff B) sym_diff C = %s = %s = A sym_diff (B sym_diff C)'%(
    (A.symmetric_difference(B)).symmetric_difference(C), 
    A.symmetric_difference((B).symmetric_difference(C))))
print('\tiii) A sym_diff B = %s = %s = A^c sym_diff B^c' %(
    A.symmetric_difference(B), (S - A).symmetric_difference(S - B) ) )
##############################################
print('properties 2: \n \t A sym_diff B = %s = %s = %s \ %s = (A U B) \ (A \cap B)'%(
    A.symmetric_difference(B), 
    ( A.union(B) ) - ( A.intersection(B) ),  
    A.union(B), A.intersection(B) ) )
##############################################
print('properties 3:')
print('\t  i) A sym_diff emptyset = %s = A'%(A.symmetric_difference(E)))
print('\t ii) A sym_diff A = %s = emptyset'%(A.symmetric_difference(A)))
##############################################
print('properties 4:')
print('\t  i) (A sym_diff B) sym_diff (B sym_diff C) = %s = %s = A sym_diff C'%(
    (A.symmetric_difference(B) ).symmetric_difference( (B.symmetric_difference(C) ) ),
    A.symmetric_difference(C)))
print('\t ii) A cap (B sym_diff C) = %s = %s = (A cap B) sym_diff (A cap C)'%(
    A.intersection( B.symmetric_difference(C) ), 
    ( A.intersection(B) ).symmetric_difference( A.intersection(C) ) ) )
##############################################


# **Remarks.** We can also use `A ^ B` to express the operation $A \, \triangle B$

# In[ ]:


print(A ^ B)
print(A.symmetric_difference(B))


# ### 5. Isdisjoint. Checking any 2 set is disjoint or not!
# 
# **Quick reminder!** 2 set $X, Y$ is called `disjoint` if $$ X \cap Y = \emptyset $$
# 
# Obviously, $A \cap \emptyset = \emptyset,$ for any set $A$

# In[ ]:


D = set([5, "6", "2"])
print('Is A and B disjoint? \t\t%s'%A.isdisjoint(B))
print('Is A and C disjoint? \t\t%s'%A.isdisjoint(C))
print('Is B and C disjoint? \t\t%s'%C.isdisjoint(B))
print('Is A and E(= emptyset) disjoint? %s'%A.isdisjoint(E))
print('Is D and A disjoint? \t\t%s'%D.isdisjoint(A))
print('Is D and B disjoint? \t\t%s'%D.isdisjoint(B))
print('Is D and C disjoint? \t\t%s'%D.isdisjoint(C))
print('Is D and E disjoint? \t\t%s'%E.isdisjoint(D))


# ### 6. Issubset & issuperset. 
# 
# Using `set_A.issubset(set_B)` to verify the given set $A$ is a subset of $B$ or not? 
# 
# In the other hand; using `set_B.issuperset(set_A)` to verify whether $B$ contains $A$ or not?

# In[ ]:


print('"A is a subset of B??". This statement is: %s.'%A.issubset(B))
print('"A is contained in S??". This statement is: %s.'%S.issuperset(A))
print('"E is a subset of B??". This statement is: %s.'%E.issubset(B))
print('"D is a subset of D??". This statement is: %s.'%D.issubset(D))
print('"D is contained in D??". This statement is: %s.'%D.issuperset(D))


# To distinguish the operator $\subset$ and $\subseteq$ in Python, we can use `<` and `<=` for any sets. For example

# In[ ]:


def is_sub_propsub(x , y):   
    """ 
        Input: x, y (set)
        return: x is subset or proper_subset or not a subset of y
    """
    if x < y:
        print('%s is a proper subset of %s'%(str(x), str(y)))
    elif x <= y:
        print('%s is a subset or identical to %s'%(str(x), str(y)))
    else:
        print('%s is not subset of %s'%(str(x), str(y)))
        
is_sub_propsub(D, D)
is_sub_propsub(D, S)
is_sub_propsub(A, S)
is_sub_propsub(A, B)


# In[ ]:


help(is_sub_propsub)


# In[ ]:


def is_sup_propsup(x , y):
    if x > y:
        print('%s is a proper superset of %s'%(str(x), str(y)))
    elif x >= y:
        print('%s is a superset or identical to %s'%(str(x), str(y)))
    else:
        print('%s is not superset of %s'%(str(x), str(y)))
        
is_sup_propsup(A, A)
is_sup_propsup(S, A)
is_sup_propsup(D, A)


# Ref: 
# - https://en.wikipedia.org/wiki/Symmetric_difference
# - https://en.wikipedia.org/wiki/Set_theory
