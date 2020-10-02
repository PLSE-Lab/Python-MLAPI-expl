# This is is a utility script for use with the Babynames exercise.
# You must have the yob2010.txt data file in your kaggle/input directory
# for this script to work. The file below can be replaced with the any of the
# data files from the Social Security dataset. With a slight modification
# to the code below, you could load all of the files at once.

# To use this script in your own notebook, you will need to add it as a new
# Kaggle Script. Create a New Notebook and in its setup dialog, set Type to: Script.
# Then just cut and paste this code into your new script file. 
# You can now import it into your Babynames notebook by selecting
# File -> Add utiliy script.

# note the dataset is stored on Kaggle in a URL that includes 2020, because of a typo!

NAMES_LIST = "/kaggle/input/yob2020/yob2010.txt"

boys = {}
girls = {}

for line in open(NAMES_LIST, 'r').readlines():
    name, gender, count = line.strip().split(",")
    count = int(count)

    if gender == "F":
        girls[name.lower()] = count
    elif gender == "M":
        boys[name.lower()] = count