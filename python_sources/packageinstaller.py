import pip
import subprocess
import sys
"""
Use to install packages outside of kaggle.
Internet must be enabled.
"""


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

'''
Examples:

install_packages = ['pyswarms', 'eli5']

for p in install_packages:
    install(p)
'''
