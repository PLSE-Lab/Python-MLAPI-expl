null=0
{"cells":[
 {
  "cell_type": "code",
  "execution_count": null,
  "metadata": {
   "collapsed": False
  },
  "outputs": [],
  "source": "# This Python 3 environment comes with many helpful analytics libraries installed\n# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python\n# For example, here's several helpful packages to load in \n\nimport numpy as np # linear algebra\nimport pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\nimport csv as csv \n\n\n# Input data files are available in the \"../input/\" directory.\n# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory\n\nfrom subprocess import check_output\nprint(check_output([\"ls\", \"../input\"]).decode(\"utf8\"))\n\n# Any results you write to the current directory are saved as output.\n\n\n# Open up the csv file in to a Python object\ncsv_file_object = csv.reader(open('../input/train.csv', 'r')) \nheader = csv_file_object.__next__()  # The next() command just skips the \n                                 # first line which is a header\ndata=[]                          # Create a variable called 'data'.\nfor row in csv_file_object:      # Run through each row in the csv file,\n    data.append(row)             # adding each row to the data variable\ndata = np.array(data) \t         # Then convert from a list to an array\n\t\t\t         # Be aware that each item is currently\n                                 # a string in this format\n        \nnumber_passangers=np.size(data[0::,1].astype(np.float))\nnumber_survived=np.sum(data[0::,1].astype(np.float))\nproportion_survived=number_passangers/number_survived\n\nwomen_only_stats=data[0::,4]=='female'\nmen_only_stats=data[0::,4]!='female'\n\nwomen_onboard=data[women_only_stats,1].astype(np.float)\nmen_onboard=data[men_only_stats,1].astype(np.float)\n\nproportion_women_survived=np.sum(women_onboard) / np.size(women_onboard)\nproportion_men_survived=np.sum(men_onboard) / np.size(men_onboard)\n\n# and then print it out\nprint('Proportion of women who survived is %s' % proportion_women_survived)\nprint('Proportion of men who survived is %s' % proportion_men_survived)\n\n\n\ntest_file = open('../input/test.csv', 'r')\ntest_file_object = csv.reader(test_file)\nheader = test_file_object.__next__()\nprediction_file = open(\"genderbasedmodel.csv\", \"w\")\nprediction_file_object = csv.writer(prediction_file)\n\nprediction_file_object.writerow([\"PassengerId\", \"Survived\"])\nfor row in test_file_object:       # For each row in test.csv\n    if row[3] == 'female':         # is it a female, if yes then                                       \n        prediction_file_object.writerow([row[0],'1'])    # predict 1\n    else:                              # or else if male,       \n        prediction_file_object.writerow([row[0],'0'])    # predict 0\ntest_file.close()\nprediction_file.close()\n"
 },
 {
  "cell_type": "code",
  "execution_count": null,
  "metadata": {
   "collapsed": False
  },
  "outputs": [],
  "source": "import seaborn\n"
 }
],"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}}, "nbformat": 4, "nbformat_minor": 0}
