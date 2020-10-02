# Some fancy Python:

import pandas as pd 
from IPython.display import HTML 

data = pd.DataFrame({"Ben": [1,2,3], "Chris": [4,5,6]})

print(HTML(data.to_html())._repr_html_())

print("\n\n\nBreak\n\n\n")

print(dir(HTML(data.to_html())))

print("\n\n\nBreak\n\n\n")

print(HTML(data.to_html()).__html__())

with open("output.html", "w") as f:
    f.write("<style>table, th, td { border: 1px solid black; border-collapse: collapse; margin: 2px}</style>")
    f.write(data.to_html(max_rows=10, index=False))
