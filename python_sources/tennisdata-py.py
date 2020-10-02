# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import sklearn

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

data = pd.read_csv("../input/atp_matches_2000.csv")

#grab only games that have left v right
games = []
games_counter = 0

for game in data: 
    winner = data["winner_hand"]
    loser = data["loser_hand"]
    
    #if game is left vs right handed
    if str(winner) != str(loser):
        games_counter = games_counter + 1
        
        #if right hand wins game
        if str(winner[1]) == "R":
            games.append(1)

#final output
print("Games with L v R: \n" + str(games_counter))
print("Games within params where Right handed player wins: \n" + str(len(games)))

#plot
x = data["winner_rank"]
y = data["minutes"]

#plt.plot(x,y,"ro")
plt.scatter(x, y)
plt.savefig("plot.png")

#right_count = 0
#left_count = 0

#for game in data['winner_hand']:
#    if game == "R":
 #       right_count = right_count + 1
  #  if game == "L":
   #     left_count = left_count + 1
    
#print(right_count)
#print(left_count)