# This script shows you how to create python functions and save
# them in an output file to use in a kernel. To see how to use them in
# a kernel go here: https://www.kaggle.com/rtatman/import-functions-from-kaggle-script/

import inspect
import os

# function that takes a number, multiplies it by two 
# and adds three
def times_two_plus_three(input):
    return (input * 2) + 3

# function that prints the word "cat"
def print_cat():
    print("cat")

# function to write the definition of our function to the file
def write_function_to_file(function, file):
    if os.path.exists(file):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not
    with open(file, append_write) as file:
        function_definition = inspect.getsource(function)
        file.write(function_definition)

# write both of our functions to our output file        
write_function_to_file(times_two_plus_three, "my_functions.py")
write_function_to_file(print_cat, "my_functions.py")