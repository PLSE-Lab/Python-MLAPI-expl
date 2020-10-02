import os
import inspect

def check_file_type(file):
    '''check for file types in directory'''
    file_type = magic.from_file(file)
    print(f"File type: \t {file_type}")
    
def check_missing_ovine(file):
    '''Check for missing sheep data'''
    if bool(re.search('lamb|ram|ewe|wether|sheep', open(file).read().lower())):
        print(f"Lamb data: \t Present")
    else:
        print(f"Lamb data: \t Missing")
        
def get_upload_date(file):
    '''Find and parse the date uploaded (only for VA files)'''
    with open(file,"r") as f:
        for line in f.readlines(): 
            if line.startswith("Richmond, VA"):
                date = dparser.parse(line,fuzzy=True).date()
                print(f"Date uploaded: \t {date}")

def check_lamb_data(directory):
    """Function to check the file type, presence of lamb data and
    header w/ Virginia USDA info for VA livestock aution reports"""
    for file in glob.glob(f"{directory}*.txt"):
        print(f"Validation info for {file}:")

        check_file_type(file)
        check_missing_ovine(file)
        get_upload_date(file)
        
        print("\n")
        

# function to write our functions to output files (easier to export)
def write_function_to_file(function, file):
    if os.path.exists(file):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not
    with open(file, append_write) as file:
        function_definition = inspect.getsource(function)
        file.write("\n")
        file.write(function_definition)

# write our functions to our output file        
write_function_to_file(check_file_type, "lamb_validation.py")
write_function_to_file(check_missing_ovine, "lamb_validation.py")
write_function_to_file(get_upload_date, "lamb_validation.py")
write_function_to_file(check_lamb_data, "lamb_validation.py")