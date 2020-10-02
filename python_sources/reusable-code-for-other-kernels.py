#
# Sometimes you want to write code once to be reused in other kernels.
#
# To export functions defined in a kernel for use in another,
# 1.  Add the "dill" custom package (supply the word 'dill' in the pip package field)
# 2.  Restart your kernel.
#

# %% imports
import dill

# %% define a bunch o' functions you want

def add(*args):
    answer = 0
    for k in args:
        answer += k
        
    return answer
    
def multiply(*args):
    answer = 1
    for k in args:
        answer *= k
        
    return answer
    
# %% export a bunch o' functions

f = open("add", "wb")
dill.dump(add,f)

f = open("multiply", "wb")
dill.dump(multiply, f)