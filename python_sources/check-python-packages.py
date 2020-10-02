# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import pip

for package in sorted(pip.get_installed_distributions(), key=lambda package: package.project_name):
    print("%-40s (%s)" %(package.project_name, package.version))
