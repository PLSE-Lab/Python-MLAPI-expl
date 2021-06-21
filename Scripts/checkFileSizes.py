import ast
from pprint import pprint
import os
import pathlib
import json


def main():
    totalErrors = 0
    count = 0
    res = {"np": [], "pd": []}
    with open("results_fs.csv", "w") as f:
        f.write("File name,File size")
        directory = os.fsencode('C:\\Users\\Kate\\Documents\\ECU\\Thesis\\python_sources_2\\')
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".py"):
                try:
                    size = str(os.path.getsize(os.path.join(directory, file)))
                    f.write("\n" + filename + "," + size)

                except Exception as e:
                    totalErrors = totalErrors + 1
    pprint(totalErrors)



if __name__ == "__main__":
    main()
