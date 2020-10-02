#!/usr/bin/env python
# coding: utf-8

# # <center> Python Good Practices: Type Hinting <center>

# Python is a dynamic programming language, i.e., the environment can be changed at run-time and isn't explicitly coded in the source code. Variables have no types, functions, and objects can be altered on run-time. Therefore, to run a python script we don't need to specify the types of the variables unlike strongly typed languages such as C++. It makes Python an easy language to learn and code. But at the same type, for debugging, documenting, improve IDEs and linters, build and maintain a cleaner architecture. and integrating with scripts written in another language, it is often useful to know the types of the variables. Python lacks it.
# To solve this issue, optional static type hinting was specified in the Python Enhancement Proposal (PEP) 484 and introduced for the first time in Python 3.5.

# ## Basic Syntax
# 
# The basic syntax of type hinting is as follows:
# 

# In[ ]:


def add_two_integers(x: int = 1, y: int = 2) -> int:
    return x + y


# Here, we use colon ```:``` to specify the type of input arguments, and arrow ```->``` to specify the type of the return variable of a function. We can use equal ```=``` to specify default value of the input parameters. 

# 
# ## Style Suggestions
# 
# - **Colon usage**: Use normal rules for colons, that is, no space before and one space after a colon (text: str).
# - **Default assignment usage**: Use spaces around the = sign when combining an argument annotation with a default value (align: bool = True).
# - **Return arrow usage**: Use spaces around the -> arrow (def headline(...) -> str).

# # Usage

# ## Basic Variables
# 
# All the following declarations are valid. Any simple basic types not mentioned below can be used.

# In[ ]:


length: int # no value at runtime until assigned
length: int = 10

is_square: bool # no value at runtime until assigned
is_square: bool = False
    
width: float # no value at runtime until assigned
width: float = 100
    
name: str # no value at runtime until assigned
name: str = "rectangle"


# ## Built-in Data Structures
# 
# To statically type built-in data structure we can use ```typing``` package as follows.

# In[ ]:


from typing import List, Set, Dict, Tuple, Optional


# ### List

# In[ ]:


x: List[int] = [1, 2]


# ### Set

# In[ ]:


x: Set[str] = {'rect', 'square'}


# ### Dictionary

# In[ ]:


x: Dict[str, float] = {'length': 10.0, 'width': 100.0}


# ### Tuples

# In[ ]:


x: Tuple[str, float, float] = ("rect", 10.0, 100.0)
    
x: Tuple[int, ...] = (1, 2, 3) # Variable size tuple


# ### Optional
# 
# Optional is used as ```Optional[A]``` to indicate that the object is either of type ```A``` or  ```None```.

# In[ ]:


def compare_numbers(x: int) -> int:
    if x<10:
        return 1
    elif x>10:
        return 0
    else:
        return None
    

x: Optional[int] = compare_numbers(10)


# ## Functions
# 

# In[ ]:


from typing import Callable, Iterator, Union


# ### Callable function
# 
# It follows the format ```Callable[[arg1, arg2], return_type]```, where ```arg1```, ```arg2``` and input types and ```return_type``` is the type of the return variable.

# In[ ]:


x: Callable[[int, int], int] = add_two_integers


# ### Iterators
# 
# For generator functions that use ```yield``` can use ```Iterators``` to specify the return type.

# In[ ]:


def generator(n: int) -> Iterator[int]:
    i = 0
    while i < n:
        yield i
        i += 1


# ### Union
# Union can be used as ```Union[A, B]``` to indicate that the object can have type ```A``` or ```B```.

# In[ ]:


def add_two_integers_or_floats(x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
    return x + y


# ## Type aliases
# 
# You can also use type aliases for the convenience of coding.

# In[ ]:


F = List[float] # Aliasing list of floats

a: F = [1.0, 4.6, 9.1]


# ## Generics
# 
# Defining a fixed type may be sometimes restrictive. We can use ```TypeVar``` for generic definitions. Here is an example from [4]

# In[ ]:


from typing import TypeVar, Iterable, DefaultDict

Relation = Tuple[T, T]
def create_tree(tuples: Iterable[Relation]) -> DefaultDict[T, List[T]]:
    tree: DefaultDict[T, List[T]] = defaultdict(list)
    for idx, child in enumerate(tuples):
        tree[idx].append(child)

    return tree

print(create_tree([(2.0,1.0), (3.0,1.0), (4.0,3.0), (1.0,6.0)]))


# # Using Mypy 

# You can use mypy python static type checker pacakge (http://mypy-lang.org/) to check whether a script has some type errors.
# 
# You can install it by running:
# 
# ```
# pip install mypy
# ```
# 
# Then you can test your script (say, ```my_script.py```) by running:
# 
# ```
# mypy my_script.py
# ```

# ## Further Reading
# 
# - https://www.kite.com/blog/python/type-hinting/
# - http://mypy-lang.org/
# - https://realpython.com/lessons/pros-and-cons-type-hints/
# - https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html
