# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
import os

print(check_output(["ls", "/kaggle/"]).decode("utf8"))
print(os.path.dirname(os.path.realpath(__file__)))
with open("/kaggle/working/__output__.json", "r") as in_file:
    txt = in_file.read()
    print (txt)

# Any results you write to the current directory are saved as output.

