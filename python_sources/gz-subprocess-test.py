
import pandas as pd
import subprocess

df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6]})

p = subprocess.Popen("gzip -c > x.csv.gz", shell=True, stdin=subprocess.PIPE, universal_newlines=True)
df.to_csv(p.stdin, chunksize=100)
p.communicate()
p.wait()
