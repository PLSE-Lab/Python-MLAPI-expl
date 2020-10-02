# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import shutil

# Download openbabel because it is not an OOB package (thank you @jmtest for the helpful import statement)
!pip install openbabel
import openbabel

# Any results you write to the current directory are saved as output.
# structure_files = os.listdir("../input/structures")

# Reads the xyz format structures with OpenBabel and converts them into a more convenient Z-Matrix format
# OpenBabel is limited in its choices for Z-Matrix so the Fenske-Hall Z-Matrix notation is used
# Fenske-Hall notation lists the number of atoms in the file on the first line followed by the element
# then first connectivity, bond distance, second connectivity, planar angle, third connectivity, last 
# is the dihedral angle
# The program creates a folder to place the converted files at "../fh_structures/"

shutil.os.mkdir("../fh_structures/")
for file in structure_files:
    converter = openbabel.OBConversion()
    converter.SetInAndOutFormats("xyz", "fh")
    molecule = openbabel.OBMol()
    converter.ReadFile(molecule, "../input/structures/" + file)
    converter.WriteFile(molecule, '../fh_structures/' + file[:-4] + ".fh")