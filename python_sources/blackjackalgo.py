# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from learntools.core import binder; binder.bind(globals())
from learntools.python.ex3 import q7 as blackjack
print('Setup complete.')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
def pivot(pivot, value):
    if value <= pivot :
        return True
    else :
        return False

def should_hit(player_total, dealer_card_val, player_aces):
    """Return True if the player should hit (request another card) given the current game
    state, or False if the player should stay. player_aces is the number of aces the player has.
    """
    if player_aces == 0:
        if dealer_card_val <= 3:
            return pivot(12,player_total)
        elif dealer_card_val <= 6:
            return pivot(11,player_total)
        else :
            return pivot(16,player_total)
    else :
        if dealer_card_val <= 8 :
            return pivot(17,player_total)
        else :
            return pivot(18,player_total)
    
#blackjack.simulate_one_game()
blackjack.simulate(n_games=1000000)

#blackjack.simulate(n_games=100000)
# Any results you write to the current directory are saved as output.