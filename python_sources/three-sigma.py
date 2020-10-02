#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system("pip install 'kaggle-environments>=0.2.1'")


# In[ ]:


import os

import multiprocessing
from joblib import Parallel, delayed

import datetime as datetime
import random

import pandas as pd


# In[ ]:





# In[ ]:





# In[ ]:


paths = [ '../input/halite-bots-c2',  
             '../input/halite-bots-c22', '../input/halite-bots-c30', '../input/halite-bots-c31',
                 '../input/halite-bots-c33']
SAVED_BOTS = []
for path in paths:
    SAVED_BOTS.extend([path + '/' + file for file in os.listdir(path) if '.py' in file])
SAVED_BOTS = sorted(SAVED_BOTS)


# In[ ]:


def printBots(bots):
    for a in [a.split('/')[-1] for a in bots]:
        print(a)


# In[ ]:


printBots(SAVED_BOTS)


# In[ ]:





# # Create Halite Environment

# In[ ]:


from kaggle_environments import evaluate, make

env = make("halite", debug=True)
env.render()


# In[ ]:





# # Current Agent

# In[ ]:





# In[ ]:





# In[ ]:


get_ipython().run_cell_magic('writefile', 'submission.py', '# for Debug/Train previous line should be commented out, uncomment to write submission.py \n\nimport numpy as np\nimport datetime as datetime\n\nprint_log = False;\nlog = []\n\ndef logit(text):\n    log.append(text)')


# In[ ]:


get_ipython().run_cell_magic('writefile', '-a submission.py', 'def reset_game_map(obs):\n    """ redefine game_map as two dimensional array of objects and set amounts of halite in each cell """\n    global game_map\n    game_map = []\n    for x in range(conf.size):\n        game_map.append([])\n        for y in range(conf.size):\n            game_map[x].append({\n                "shipyard": None,\n                "ship": None,\n                "halite": obs.halite[conf.size * y + x],\n                "targeted": 0,\n                \'enemy_halite\': 0,\n                \n                "lightest_enemy_touching": 1e6,\n                \'enemies_touching\': 0,\n                \'enemy_halite_touching\': 0,\n                \n                \'lightest_enemy_nearby\': 1e6,\n                \'enemies_nearby\': 0,\n                \'enemy_halite_nearby\': 0, \n                \n                \'lightest_enemy_within_3\': 1e6,\n                \'enemies_within_3\': 0,\n                \'enemy_halite_within_3\': 0, \n                \n                "please_move": 0,\n                "touching_base": False,\n            })\n\ndef get_my_units_coords_and_update_game_map(obs):\n    """ get lists of coords of my units and update locations of ships and shipyards on the map """\n    # arrays of (x, y) coords\n    global game_map\n    my_shipyards_coords = []\n    my_ships_coords = []\n    \n    for player in range(len(obs.players)):\n        shipyards = list(obs.players[player][1].values())\n        for shipyard in shipyards:\n            x = shipyard % conf.size\n            y = shipyard // conf.size\n            # place shipyard on the map\n            game_map[x][y]["shipyard"] = player\n            if player == obs.player:\n                my_shipyards_coords.append((x, y))\n                for spot in [(x, y), (c(x-1), y), (c(x+1), y), (x, c(y-1)), (x, c(y+1))]:\n                    game_map[spot[0]][spot[1]]["touching_base"] = True\n        \n        ships = list(obs.players[player][2].values())\n        for ship in ships:\n            x = ship[0] % conf.size\n            y = ship[0] // conf.size\n            # place ship on the map\n            game_map[x][y]["ship"] = player\n            if player == obs.player: # mine\n                my_ships_coords.append((x, y))\n            else: # enemy ship\n                game_map[x][y]["enemy_halite"] = ship[1]\n                for spot in [(x, y), (c(x-1), y), (c(x+1), y), (x, c(y-1)), (x, c(y+1))]:\n                    \n                    game_map[spot[0]][spot[1]]["enemies_touching"] += 1\n                    game_map[spot[0]][spot[1]]["enemy_halite_touching"] += ship[1]\n                    \n                    if ship[1] < game_map[spot[0]][spot[1]]["lightest_enemy_touching"]:\n                        game_map[spot[0]][spot[1]]["lightest_enemy_touching"] = ship[1]\n                \n                for spot in [(x, y), \n                              (c(x-1), y), (c(x+1), y), (x, c(y-1)), (x, c(y+1)),\n                               (c(x-2), y), (c(x+2), y), (x, c(y-2)), (x, c(y+2)),\n                                    (c(x-1), c(y-1)), (c(x-1), c(y+1)), (c(x+1), c(y-1)),  (c(x+1), c(y+1)),]:\n                    \n                    game_map[spot[0]][spot[1]]["enemies_nearby"] += 1\n                    game_map[spot[0]][spot[1]]["enemy_halite_nearby"] += ship[1]\n                    \n                    if ship[1] < game_map[spot[0]][spot[1]]["lightest_enemy_nearby"]:\n                        game_map[spot[0]][spot[1]]["lightest_enemy_nearby"] = ship[1]\n                    \n                for spot in [(x, y), \n                              (c(x-1), y), (c(x+1), y), (x, c(y-1)), (x, c(y+1)),\n                               (c(x-2), y), (c(x+2), y), (x, c(y-2)), (x, c(y+2)),\n                                    (c(x-1), c(y-1)), (c(x-1), c(y+1)), (c(x+1), c(y-1)),  (c(x+1), c(y+1)),\n                               (c(x-3), y), (c(x+3), y), (x, c(y-3)), (x, c(y+3)),\n                                  (c(x-2), c(y-1)), (c(x-2), c(y-1)), (c(x+2), c(y-1)),  (c(x+2), c(y+1)),\n                                  (c(x-1), c(y-2)), (c(x-1), c(y+2)), (c(x+1), c(y-2)),  (c(x+1), c(y+2)),\n                            ]:\n                    \n                    if ship[1] < game_map[spot[0]][spot[1]]["lightest_enemy_within_3"]:\n                        game_map[spot[0]][spot[1]]["lightest_enemy_within_3"] = ship[1]\n                        \n                    game_map[spot[0]][spot[1]]["enemies_within_3"] += 1\n                    game_map[spot[0]][spot[1]]["enemy_halite_within_3"] += ship[1]    \n    \n    return my_shipyards_coords, my_ships_coords')


# In[ ]:


get_ipython().run_cell_magic('writefile', '-a submission.py', '\ndef get_x(x):\n    return (x % conf.size)\n\ndef get_y(y):\n    return (y % conf.size)\n\ndef c(s):\n    return (s % conf.size)\n\ndef enemy_ship_near(x, y, player, ship_halite):\n    """ check if enemy ship can attack a square """\n#     for spot in [game_map[x][y], game_map[c(x-1)][y], game_map[c(x+1)][y], game_map[x][c(y-1)], game_map[x][c(y+1)] ]:\n#         if (spot["ship"] != None and spot["ship"] != player \n#                      and spot["enemy_halite"] <= ship_halite): \n#             return True\n#     return False\n    if game_map[x][y]["lightest_enemy_touching"] <= ship_halite:\n        return True\n    else:\n        return False\n\ndef threat(x, y, player, ship_halite):\n    spot = game_map[get_x(x)][get_y(y)]\n    return ( (spot["ship"] != None and spot["ship"] != player \n                     and spot["enemy_halite"] <= ship_halite) \n             or (spot["shipyard"] != None and spot["shipyard"] != player ) )\n    \n\ndef nearestThreateningShip(x, y, player, ship_halite):\n    for dist in range(conf.size):\n        for xs in range(dist + 1):\n            ys = dist-xs;\n            if ( threat(x + xs, y + ys, player, ship_halite) or\n                 threat(x - xs, y + ys, player, ship_halite) or\n                 threat(x + xs, y - ys, player, ship_halite) or\n                 threat(x - xs, y - ys, player, ship_halite) ):\n                logit(\'nearest threat is {} units away\'.format(dist))\n                return dist;\n            \n    else:\n        logit(\'no threats?\')\n        return 100;')


# In[ ]:


get_ipython().run_cell_magic('writefile', '-a submission.py', '\n# def isBase(x, y, player):\n#     if game_map[x][y]["shipyard"] != None and game_map[x][y]["shipyard"] == player:\n#         return True\n#     else:\n#         return False\n    \n# def touchingBase(x, y, player):\n#     for spot in [game_map[x][y], game_map[c(x-1)][y], game_map[c(x+1)][y], game_map[x][c(y-1)], game_map[x][c(y+1)] ]:\n#         if spot["shipyard"] != None and spot["shipyard"] == player:\n#             return True\n#     return False\n\n\'\'\'check whether current spot and the four spots can move to are all threatened\'\'\'\ndef surrounded(x, y, player, halite_onboard):\n    for spot in [(x, y), (c(x-1), y), (c(x+1), y), (x, c(y-1)), (x, c(y+1))]:\n        if not enemy_ship_near(*spot, player, halite_onboard):\n            return False\n    return True\n\n\'\'\'check if have only one possible move\'\'\'\ndef onlyOneOption(x, y, player, halite_onboard, will_mine, moveable = False):\n    viable_spot = []\n    if clearArea(x, y, player, halite_onboard + will_mine, moveable):\n        viable_spot = (x, y) \n    \n    for spot in [(c(x-1), y), (c(x+1), y), (x, c(y-1)), (x, c(y+1))]:\n        if clearArea(*spot, player, halite_onboard, moveable):\n            if len(viable_spot) > 0: # if have multiple spots\n                return None\n            else:\n                viable_spot = spot\n    if len(viable_spot) == 0:\n        logit("should have been flagged as surrounded")\n        return None\n        \n    return viable_spot\n\ndef clear(x, y, player, halite_onboard, moveable = False):\n    """ check if cell is clear to move into """\n    if ((game_map[x][y]["shipyard"] == player or game_map[x][y]["shipyard"] == None) and\n            ( (game_map[x][y]["ship"] == player and moveable)   or   \n               ( game_map[x][y]["ship"] == None )  or \n              ( game_map[x][y]["ship"] !=  player  and game_map[x][y]["enemy_halite"] > halite_onboard)) ):\n        return True\n    return False\n\ndef clearArea(x, y, player, halite_onboard, moveable = False):\n    return (clear(x, y, player, halite_onboard, moveable) and not enemy_ship_near(x, y, player, halite_onboard))')


# In[ ]:


# %%writefile -a submission.py
# def mustCrossBase(x, y, x_dir, y_dir, player):
#     if x_dir == 0 and isBase(x, c(y + y_dir), player):
#         return True
#     elif y_dir == 0 and isBase(c(x + x_dir), y, player):
#         return True
#     else:
#         return False


# In[ ]:


get_ipython().run_cell_magic('writefile', '-a submission.py', '# def distanceTo(xf, yf, x_pos, y_pos):\n#     return np.min( (   (xf - x_pos) % conf.size, conf.size - ((xf - x_pos) % conf.size) ) ) \\\n#             + np.min( (   (yf - y_pos) % conf.size, conf.size - ((yf - y_pos) % conf.size) ) ) \n\ndef distanceTo(xf, yf, x_pos, y_pos):\n    xd = abs(xf - x_pos)\n    yd = abs(yf - y_pos)\n    return min(xd, 21-xd) + min(yd, 21-yd)\n\ndef directionTo(xf, yf, x_pos, y_pos):\n    return (0 if xf==x_pos else (1 if (xf - x_pos) % conf.size <= conf.size // 2 else -1)), \\\n              (0 if yf==y_pos else (1 if (yf - y_pos) % conf.size <= conf.size // 2 else -1)),')


# In[ ]:


get_ipython().run_cell_magic('writefile', '-a submission.py', '\n# ship values \nSHIP_MINE_PCT = 0.2  \nSHIP_HUNT_PCT = 0.10     \nAVG_HALITE_PWR = 0.8\nENEMY_HALITE_PWR = 0.8\n\n# final drop-offs\nMIN_FINAL_DROPOFF = 100   \nPCTILE_DROPOFF = 95   \nRETURN_HOME = 360  \nMUST_DROPOFF = 2000\n\n# mining variables\nEXPECTED_MINE_TURNS = 3     # possibly lower//\nEXPECTED_MINE_FRAC = np.sum([ 0.25 * (1 - 0.25 + 0.02) ** i for i in range(int(EXPECTED_MINE_TURNS))])\n\nINERTIA = 1.1   \nDISTANCE_DECAY = 0.06  \n\n# return and conversion variables\nRETURN_CHANCE = 0.5                        \nYARD_CONVERSION_CHANCE = 0.25   \nBASE_YARDS = 0.5 \nSHIP_TO_BASE_MULT = 2.5    \n\n# raid variables\nRAID_BASE_TURNS = 1       # WILDCARD -- may also generate a lot of variety\nRAID_DISTANCE_DECAY = 0.15\nRAID_RETARGET_DECAY = 0.35  \n\n# deprecated\nRETURN_RESIDUAL = 0 \nMOVE_MULTIPLIER = 1  \nRAID_ONBOARD_PENALTY = 0\nRAID_MINE_RESIDUAL = 0    \n\n\n# raid parameters - enemies and halite nearby\nRAID_MULTIPLIER = 0.10 * 1.5      # likely increase this//\nREHT_MULTIPLIER = 0.08 * 1.5\nREHN_MULTIPLIER = 0.03   \nREH3_MULTIPLIER = 0.015 \n\nRLEN_PENALTY = -60     # been moving higher\nRLE3_PENALTY = -30  \n\n# mining parameters - nearness to base\nMINE_BASE_DIST_MULT = 0.0\n\n# mining parameters - enemies nearby\nMET_ADJUSTMENT = 0\nMEN_ADJUSTMENT = 0\nME3_ADJUSTMENT = 0\n\nMLET_FWD = 0.25\nMLEN_FWD = 0.25\nMLE3_FWD = 0.25\n\nMLET_PENALTY = -150     # probably slash this //\nMLEN_PENALTY = -10 \nMLE3_PENALTY = 0\n\nMEHT_BONUS = 0\nMEHN_BONUS = 0\nMEH3_BONUS = 0\n\nMM_GAIN = 0.25 * 0\nSUR_GAIN = 0.25 * 0\nOOO_GAIN = 0.25 * 0 ')


# In[ ]:


get_ipython().run_cell_magic('writefile', '-a submission.py', '\ndef findBestSpot(x_pos, y_pos, player, my_halite, halite_onboard, \n                 my_shipyards_coords, num_shipyards, avg_halite, step, num_ships,\n                   live_run = False):   \n    moveable = ~live_run\n    \n    must_move = enemy_ship_near(x_pos, y_pos, player, \n                                  halite_onboard  + MM_GAIN * game_map[x_pos][y_pos]["halite"]  )  # ***\n    \n    ship_inertia = (INERTIA if not game_map[x_pos][y_pos]["touching_base"] else 1)\n    \n    current_halite = game_map[x_pos][y_pos]["halite"]\n    best_spot = (x_pos, y_pos, np.floor(current_halite \\\n                                            * EXPECTED_MINE_FRAC / EXPECTED_MINE_TURNS \\\n                                             * (ship_inertia) \n                                              * (0 if must_move else 1) ), \'remain\')\n    \n    # check if surrounded\n    if surrounded(x_pos, y_pos, player, halite_onboard, SUR_GAIN * game_map[x_pos][y_pos]["halite"] ):\n        if halite_onboard > conf.convertCost or my_halite > conf.convertCost:\n            logit(\'emergency conversion at ({}, {}) to preserve {} halite\'.format(x_pos, y_pos, halite_onboard))\n            return (x_pos, y_pos, np.floor(halite_onboard) + conf.convertCost, \'conversion\') \n        else:\n            logit(\'surrounded at ({}, {}) but not enough cash for emergency conversion\'.format(x_pos, y_pos))\n    \n    # check if only one option\n    ooo = onlyOneOption(x_pos, y_pos, player, halite_onboard, OOO_GAIN * game_map[x_pos][y_pos]["halite"],  moveable)\n    if ooo is not None:\n        logit(\'only one option from ({}, {}), heading to {}, {}\'.format(x_pos, y_pos, ooo[0], ooo[1]))\n        return ( ooo[0], ooo[1], np.floor(halite_onboard) + conf.spawnCost, \'only one option\' )\n    \n    if halite_onboard > conf.convertCost:\n        closest_yard, min_dist = findNearestYard(my_shipyards_coords, x_pos, y_pos)\n        spot_value = halite_onboard * YARD_CONVERSION_CHANCE * min_dist / 20 \\\n                       * (1 / ( BASE_YARDS + num_shipyards ) )  #* (50 / (avg_halite + BASE_AVG_HALITE))\n        if spot_value > best_spot[2]:\n            best_spot = (x_pos, y_pos, np.floor(spot_value), \'conversion\')\n    \n    ship_value = getShipValue(num_ships, step)\n    return_penalty = ( (ship_value - conf.spawnCost) \n                         if ( my_halite > conf.spawnCost and ship_value > conf.spawnCost ) else 0)\n    \n    # consider returning\n    return_value = 0; nearest_base = 20;\n    for shipyard_coords in my_shipyards_coords:\n        x, y = shipyard_coords\n        x_dir, y_dir = directionTo(x, y, x_pos, y_pos)\n         \n        # if dangerous spot or both paths are dangerous, ignore;\n        if ( not clearArea(x, y, player, halite_onboard, moveable) or\n            ( not clearArea( (x_pos + x_dir) % conf.size, y_pos, player, halite_onboard, moveable)  and  \n               not clearArea( x_pos, (y_pos + y_dir) % conf.size, player, halite_onboard, moveable)) ):\n            continue;\n\n        \n        dist = distanceTo(x, y, x_pos, y_pos) #  * (step + base_step) ** 2 / (base_step + 400)\n        \n        if dist < nearest_base:\n            nearest_base = dist\n        \n        spot_value = ( halite_onboard * RETURN_CHANCE  \n                        / ( dist * MOVE_MULTIPLIER + 1e-6) \n                             * np.min( ( np.sqrt(step/ 30), 1 ) )\n                             * np.clip( ( ship_value / conf.spawnCost), 1, 1.5)         \n                      )\n        if spot_value > return_value:\n            return_value = spot_value\n        if spot_value > best_spot[2] and not enemy_ship_near(x, y, player, halite_onboard):\n            best_spot = (x, y, np.floor(spot_value), \'dropoff\')\n\n    for x in range(conf.size):\n        for y in range(conf.size):\n            x_dir, y_dir = directionTo(x, y, x_pos, y_pos)\n             \n            # if dangerous spot or both paths are dangerous, ignore;\n            if ( not clearArea(x, y, player, halite_onboard, moveable) or\n                ( not clearArea( c(x_pos + x_dir), y_pos, player, halite_onboard, moveable)  and  \n                   not clearArea( x_pos, c(y_pos + y_dir), player, halite_onboard, moveable)) ):\n                continue;\n            \n\n                  \n                    \n                    \n                    \n                    \n            # Consider Raiding: \n            spot_raid_value = 0\n            if game_map[x][y]["enemy_halite"] > halite_onboard:\n                dist = distanceTo(x, y, x_pos, y_pos)\n                spot_raid_value = ( (game_map[x][y]["enemy_halite"] * RAID_MULTIPLIER  \n                                - RAID_ONBOARD_PENALTY * halite_onboard  \n                                    \n                                     + game_map[x][y][\'enemy_halite_touching\'] * REHT_MULTIPLIER\n                                     + game_map[x][y][\'enemy_halite_nearby\'] * REHN_MULTIPLIER\n                                     + game_map[x][y][\'enemy_halite_within_3\'] * REH3_MULTIPLIER\n                                     \n                                     + (RLEN_PENALTY if game_map[x][y][\'lightest_enemy_nearby\'] < halite_onboard \n                                                             else 0) \n \n                                     + (RLE3_PENALTY if game_map[x][y][\'lightest_enemy_within_3\'] < halite_onboard \n                                                             else 0) \n \n                                     \n                                    )\n                             \n                            / ( RAID_BASE_TURNS + dist * MOVE_MULTIPLIER )   \n                                    * np.exp( - RAID_DISTANCE_DECAY * dist )  \n                               * np.exp( - RAID_RETARGET_DECAY * game_map[x][y]["targeted"] ) \n                                  )\n                    \n                if spot_raid_value > best_spot[2]:\n                    best_spot = (x, y, np.floor(spot_raid_value), \'raiding\')\n            \n            \n            \n            # Mining: consider if must move or beats current spot, and not yet targeted\n            if  ( ( must_move or game_map[x][y]["halite"] > current_halite * ship_inertia )\n                    and game_map[x][y]["targeted"] == 0):\n                dist = distanceTo(x, y, x_pos, y_pos)\n                spot_value = ( game_map[x][y]["halite"] * EXPECTED_MINE_FRAC \n                            / ( EXPECTED_MINE_TURNS + dist * MOVE_MULTIPLIER \n                                                     + nearest_base * MINE_BASE_DIST_MULT )  \n                                    * np.exp( - DISTANCE_DECAY * dist)\n                                       * (0.7 if game_map[x][y]["touching_base"] else 1)\n                              +  ( (return_value * RETURN_RESIDUAL - return_penalty) \n                                           if game_map[x][y]["touching_base"] else 0)\n                              \n                              + game_map[x][y]["enemies_touching"] * MET_ADJUSTMENT\n                              + game_map[x][y]["enemies_nearby"] * MEN_ADJUSTMENT \n                              + game_map[x][y]["enemies_within_3"] * ME3_ADJUSTMENT\n                                     \n                              + game_map[x][y]["enemy_halite_touching"] * MEHT_BONUS\n                              + game_map[x][y]["enemy_halite_nearby"] * MEHN_BONUS \n                              + game_map[x][y]["enemy_halite_within_3"] * MEH3_BONUS\n                      \n                              + (MLET_PENALTY if (game_map[x][y][\'lightest_enemy_touching\'] \n                                                    < halite_onboard + game_map[x][y]["halite"] * MLET_FWD)\n                                                     else 0) \n                              + (MLEN_PENALTY if (game_map[x][y][\'lightest_enemy_nearby\'] \n                                                    < halite_onboard + game_map[x][y]["halite"] * MLEN_FWD)\n                                                     else 0) \n\n                              + (MLE3_PENALTY if (game_map[x][y][\'lightest_enemy_within_3\'] \n                                                    < halite_onboard + game_map[x][y]["halite"] * MLE3_FWD)\n                                                     else 0) \n \n                              + spot_raid_value * RAID_MINE_RESIDUAL)\n                \n                \n                if spot_value > best_spot[2]:\n                    best_spot = (x, y, np.floor(spot_value), \'mining\')\n\n   \n    if best_spot[0] == x_pos and best_spot[1] == y_pos: # staying put\n        sp = True\n    else:  # if moving;\n        if live_run:\n            game_map[best_spot[0]][best_spot[1]]["targeted"] += 1\n\n    if best_spot[3] == \'dropoff\':\n        logit(\'   moving from ({}, {}) to ({}, {}) to dropoff {} halite\'.format(\n            x_pos, y_pos, best_spot[0], best_spot[1], halite_onboard))\n    if best_spot[3] == \'raiding\':\n        logit(\'attempting raid of ({}, {}) from ({}, {}) to gain {} halite\'.format(\n            best_spot[0], best_spot[1], x_pos, y_pos, game_map[best_spot[0]][best_spot[1]]["enemy_halite"] ))\n        \n    \n    \n    return best_spot')


# In[ ]:


get_ipython().run_cell_magic('writefile', '-a submission.py', '\ndef findNearestYard(my_shipyards_coords, x, y):\n    """ find nearest shipyard to deposit there"""\n    min_dist = conf.size * 2\n    closest_yard = 0\n    for yard_idx, yard in enumerate(my_shipyards_coords):\n        dist = np.min( (  ((x - my_shipyards_coords[yard_idx][0]) % conf.size), \n                                (21 - ((x - my_shipyards_coords[yard_idx][0]) % conf.size))  ) ) \\\n          + np.min( (  ((y - my_shipyards_coords[yard_idx][1]) % conf.size), \n                     (21 - ((y - my_shipyards_coords[yard_idx][1]) % conf.size))  ) )\n        if dist < min_dist:\n            min_dist = dist;\n            closest_yard = yard_idx\n    return closest_yard, min_dist\n            ')


# In[ ]:


get_ipython().run_cell_magic('writefile', '-a submission.py', 'def moveTo(x_initial, y_initial, x_target, y_target, ship_id, halite_onboard, player, actions):\n    """ move toward target as quickly as possible without collision (or later, bad collision)"""\n    if (x_target - x_initial) % conf.size <=  ( 1 + conf.size) // 2 :\n        # move down\n        x_dir = 1;\n        x_dist = (x_target - x_initial) % conf.size\n    else:\n        # move up\n        x_dir = -1;\n        x_dist = (x_initial - x_target) % conf.size\n    \n    if (y_target - y_initial) % conf.size <=  ( 1 + conf.size) // 2 :\n        # move down\n        y_dir = 1;\n        y_dist = (y_target - y_initial) % conf.size\n    else:\n        # move up\n        y_dir = -1;\n        y_dist = (y_initial - y_target) % conf.size\n    \n    action = None\n    if x_dist > y_dist:\n        # move X first if can;\n        if clearArea( ( x_initial + x_dir) % conf.size, y_initial, player, halite_onboard):\n            action = (\'WEST\' if x_dir <0 else \'EAST\')\n        elif clearArea( x_initial, ( y_initial + y_dir) % conf.size, player, halite_onboard) :\n            action = (\'NORTH\' if y_dir < 0 else \'SOUTH\')\n        \n    else:\n        # move Y first if can\n        if clearArea( x_initial, ( y_initial + y_dir) % conf.size, player, halite_onboard) :\n            action = (\'NORTH\' if y_dir < 0 else \'SOUTH\')\n        elif clearArea( ( x_initial + x_dir) % conf.size, y_initial, player, halite_onboard):\n            action = (\'WEST\' if x_dir <0 else \'EAST\')\n    \n    # if area was not clear, then just move whoever is currently empty\n    if enemy_ship_near(x_initial, y_initial, player, halite_onboard):\n        logit(\'moving into traffic from ({}, {})\'.format(x_initial, y_initial))\n        if x_dist > y_dist:\n            # move X first if can;\n            if clear( ( x_initial + x_dir) % conf.size, y_initial, player, halite_onboard):\n                action = (\'WEST\' if x_dir <0 else \'EAST\')\n            elif clear( x_initial, ( y_initial + y_dir) % conf.size, player, halite_onboard) :\n                action = (\'NORTH\' if y_dir < 0 else \'SOUTH\')\n\n        else:\n            # move Y first if can\n            if clear( x_initial, ( y_initial + y_dir) % conf.size, player, halite_onboard) :\n                action = (\'NORTH\' if y_dir < 0 else \'SOUTH\')\n            elif clear( ( x_initial + x_dir) % conf.size, y_initial, player, halite_onboard):\n                action = (\'WEST\' if x_dir <0 else \'EAST\')\n            \n    if action is not None:\n        game_map[x_initial][y_initial]["ship"] = None\n        actions[ship_id] = action\n        \n    if action == \'NORTH\':\n        game_map[x_initial][c(y_initial - 1)]["ship"] = player\n    elif action == \'SOUTH\':\n        game_map[x_initial][c(y_initial + 1)]["ship"] = player\n    elif action == \'WEST\':\n        game_map[c(x_initial - 1)][y_initial]["ship"] = player\n    elif action == \'EAST\':\n        game_map[c(x_initial + 1)][y_initial]["ship"] = player\n    \n    \n    return actions\n\n# def get_directions(i0, i1, i2, i3):\n#     """ get list of directions in a certain sequence """\n#     return [directions[i0], directions[i1], directions[i2], directions[i3]]')


# In[ ]:


get_ipython().run_cell_magic('writefile', '-a submission.py', '\ndef orderBasesForSpawning(shipyards_keys, my_shipyards_coords, player, my_ships_coords):\n    base_values = np.zeros(len(my_shipyards_coords))\n    for yard_idx, yard_coords in enumerate(my_shipyards_coords):\n        x = yard_coords[0]\n        y = yard_coords[1]\n        \n        # ensure base is clear\n        if not clear(x, y, player, 0):\n            continue;\n        \n        nearby_halite = 0\n        nearby_enemy_halite = 0\n        nearby_ships = 0.1\n        \n        BASE_MINING_TURNS = 3\n        BASE_SHIP_TURNS = 0.5\n        RADIUS = 10\n        ENEMY_DISTANCE_DECAY = 0.10\n        \n        BASE_SHIPS_PWR = 0\n        \n        RAIDING_EFFICIENCY = 0.03  \n        \n        # look at halite nearby (favor lots of halite nearby)\n        for xs in range(-RADIUS, RADIUS + 1):\n            for ys in range(-RADIUS, RADIUS + 1):\n                nearby_halite += ( game_map[c(x + xs)][c(y + ys)]["halite"] \n                                            / (BASE_MINING_TURNS + 2 * ( abs(xs) + abs(ys) ) ) )\n                nearby_enemy_halite += ( game_map[c(x + xs)][c(y + ys)]["enemy_halite"] \n                                                  * np.exp( - ENEMY_DISTANCE_DECAY * ( abs(xs) + abs(ys) ) ) )\n        \n        for s in my_ships_coords:\n            nearby_ships +=  ( 1 / (BASE_SHIP_TURNS + distanceTo( s[0], s[1], x, y) ) )\n            \n        \n        base_values[yard_idx] = (  ( nearby_halite + RAIDING_EFFICIENCY * nearby_enemy_halite ) \n                                       / ( nearby_ships ** BASE_SHIPS_PWR ) ) \n        \n    priorities = np.argsort(base_values)[::-1]\n    logit( [int(base_values[i]) for i in priorities])\n    \n    my_shipyards_coords = [ my_shipyards_coords[i] for i in priorities]\n    shipyards_keys = [shipyards_keys[i] for i in priorities] \n    \n    return shipyards_keys, my_shipyards_coords')


# In[ ]:


get_ipython().run_cell_magic('writefile', '-a submission.py', '\ndef getAverageHalite():\n    total_halite = 0\n    for x in range(conf.size):\n        for y in range(conf.size):\n            total_halite += game_map[x][y]["halite"]\n    return total_halite / (conf.size ** 2)\n    \n\ndef getAverageEnemyHalite():\n    total_enemy_halite = 0\n    for x in range(conf.size):\n        for y in range(conf.size):\n            total_enemy_halite += game_map[x][y]["enemy_halite"]\n    return total_enemy_halite / (conf.size ** 2)\n\ndef define_some_globals(config):\n    """ define some of the global variables """\n    global conf\n    global convert_plus_spawn_cost\n    global globals_not_defined\n    conf = config\n    convert_plus_spawn_cost = conf.convertCost + conf.spawnCost\n    globals_not_defined = False\n\n\n############################################################################\nconf = None\ngame_map = [] \nconvert_plus_spawn_cost = None  \n \nglobals_not_defined = True \n\ndef getShipValue(num_ships, step):\n    return (  (SHIP_MINE_PCT * getAverageHalite() ** AVG_HALITE_PWR  +\n                            SHIP_HUNT_PCT * getAverageEnemyHalite() ** ENEMY_HALITE_PWR ) \n                            * (10 / num_ships)\n                              * (400 - step) \n            \n            + (1000 if num_ships < 3 else 0)\n            + (3000 if num_ships < 2 else 0)\n           \n           ) \n\n\n    \ndef strategicShipSpawning(my_halite, actions, num_ships, shipyards_keys, my_shipyards_coords, step, player):\n    for i in range(len(my_shipyards_coords)):\n        ship_value  =  getShipValue(num_ships, step)\n        logit(\'Ship Value: {:.0f}\'.format(ship_value))\n        \n        if (my_halite < conf.spawnCost):\n            logit(\'  not enough halite to spawn ship\')\n            break;\n            \n        if ship_value < conf.spawnCost:\n            break;\n\n        x = my_shipyards_coords[i][0]\n        y = my_shipyards_coords[i][1]\n        if clear(x, y, player, 0):\n            my_halite -= conf.spawnCost\n            actions[shipyards_keys[i]] = "SPAWN"\n            num_ships += 1\n            game_map[x][y]["ship"] = player\n            logit(\'spawning new ship at ({}, {})\'.format(x, y))\n\n                \n    return my_halite, actions')


# In[ ]:


get_ipython().run_cell_magic('writefile', '-a submission.py', '\n\ndef timeout(start_time, player, step, rd, ship, config):\n\n    TIMEOUT = 0.7 * config.actTimeout *  1e6   # * 100e3  \n    \n    if (datetime.datetime.now() - start_time).microseconds > TIMEOUT:\n        print(\'AGENT {} TIMED OUT ON STEP {} - for Round {} and ship number {}\'.format(player, step, rd, ship))\n        return True;\n    else:\n        return False;\n\ndef swarm_agent(obs, config):\n    start_time = datetime.datetime.now()\n    logit(\'\\nStep {}\'.format(obs.step))\n    if globals_not_defined:\n        define_some_globals(config)\n    actions = {}\n    my_halite = obs.players[obs.player][0]\n    \n    reset_game_map(obs)\n    my_shipyards_coords, my_ships_coords = get_my_units_coords_and_update_game_map(obs)\n    num_shipyards = len(my_shipyards_coords)\n    \n    ships_keys = list(obs.players[obs.player][2].keys()); ships_values = list(obs.players[obs.player][2].values())\n    shipyards_keys = list(obs.players[obs.player][1].keys())\n    avg_halite = getAverageHalite()\n\n    # order the actions of ships\n    move_values = []; # moves = []\n    for i in range(len(my_ships_coords)):\n        if timeout(start_time, obs.player, obs.step, 1, i, config):\n            return actions;\n        halite_onboard = ships_values[i][1]\n        x = my_ships_coords[i][0]; y = my_ships_coords[i][1]\n        \n        x_target, y_target, spot_value, purpose = findBestSpot(x, y, obs.player, my_halite, halite_onboard,\n                                                                          my_shipyards_coords, num_shipyards,\n                                                                         avg_halite, obs.step, len(ships_keys), \n                                                                           False)\n        x_dir, y_dir = directionTo(x_target, y_target, x, y)\n#         moves.append( c(x + x_dir), c(y + y_dir) )\n        if spot_value > game_map[c(x + x_dir)][c(y + y_dir)]["please_move"]:\n            game_map[c(x + x_dir)][c(y + y_dir)]["please_move"] = spot_value + 1\n            \n        if purpose == \'only one option\':\n            logit(\'only one option for ship at ({}, {}) with {} halite onboard\'.format(x, y, halite_onboard))\n        move_values.append(int(spot_value))\n    \n    # check for any \'please move\' requests\n    for i in range(len(my_ships_coords)):\n        x = my_ships_coords[i][0]; y = my_ships_coords[i][1]\n        if game_map[x][y]["please_move"] > move_values[i]:\n            move_values[i] = game_map[x][y]["please_move"]\n            logit("please move off ({}, {}) with weight {}".format(x, y, move_values[i]))\n      \n    priorities = np.argsort(move_values)[::-1]\n    logit( [move_values[i] for i in priorities])\n    \n    my_ships_coords = [my_ships_coords[i] for i in priorities]\n    ships_keys = [ships_keys[i] for i in priorities]\n    ships_values = [ships_values[i] for i in priorities]\n    \n    logit(\'\\n - actions - \\n\')\n        \n    # execute ship actions\n    for i in range(len(my_ships_coords)):\n        if timeout(start_time, obs.player, obs.step, 2, i, config):\n            return actions;\n        \n        halite_onboard = ships_values[i][1]\n        x = my_ships_coords[i][0]; y = my_ships_coords[i][1]\n\n        # no yards or enormous halite surplus, must convert;\n        if ( (num_shipyards==0 and (my_halite >= conf.convertCost or halite_onboard >= convert_plus_spawn_cost))\n               or (halite_onboard >= convert_plus_spawn_cost * SHIP_TO_BASE_MULT) ):\n            actions[ships_keys[i]] = "CONVERT"\n            num_shipyards += 1\n            logit(\'forced conversion\')\n            continue;\n           \n        # \'\'\'final dropoff or must dropoff\'\'\'\n        elif ( (obs.step > RETURN_HOME) and \n              (  (halite_onboard > MIN_FINAL_DROPOFF) or \n                       (halite_onboard >  np.percentile( [s[1] for s in ships_values], PCTILE_DROPOFF) ) )\n           or    (halite_onboard > MUST_DROPOFF  ) ):\n            \n            if len(my_shipyards_coords) > 0:\n                closest_yard, min_dist = findNearestYard(my_shipyards_coords, x, y)\n                actions = moveTo(x, y, *my_shipyards_coords[closest_yard], ships_keys[i], halite_onboard,\n                                         obs.player, actions)\n                \n        # \'\'\'figure out best move\'\'\' \n        else:\n            x_target, y_target, new_spot_halite, purpose = findBestSpot(x, y, obs.player, my_halite, halite_onboard,\n                                                                          my_shipyards_coords, num_shipyards,\n                                                                avg_halite, obs.step, len(ships_keys), True)\n            if not (x_target == x and y_target == y):\n#                 logit(\'aiming to get to ({}, {}) from ({}, {})\'.format(x_target,y_target, x ,y))\n                actions = moveTo(x, y, x_target, y_target, ships_keys[i], halite_onboard, obs.player, actions)\n            elif purpose == \'conversion\':\n                actions[ships_keys[i]] = "CONVERT"\n                logit(\'converting to base\')\n                num_shipyards += 1\n\n    if my_halite >= conf.spawnCost:\n        shipyards_keys, my_shipyards_coords = orderBasesForSpawning(shipyards_keys, my_shipyards_coords,\n                                                                       obs.player, my_ships_coords )\n\n    # YARDS AND SPAWNING:\n    # auto-spawn if no ships, or first 30 turns\n    if ( len(ships_keys) == 0 or obs.step <= 30) and my_halite >= conf.spawnCost:\n        for i in range(len(my_shipyards_coords)):\n            if my_halite >= conf.spawnCost:\n                x = my_shipyards_coords[i][0]\n                y = my_shipyards_coords[i][1]\n                if clear(x, y, obs.player, 0):\n                    my_halite -= conf.spawnCost\n                    actions[shipyards_keys[i]] = "SPAWN"\n                    game_map[x][y]["ship"] = obs.player\n                    logit(\'auto-spawn\')\n\n    # strategic spawning:\n    else:\n        my_halite, actions = strategicShipSpawning(my_halite, actions, len(ships_keys),\n                                                   shipyards_keys, my_shipyards_coords, \n                                                   obs.step, obs.player)\n        \n    return actions')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Agent Selection

# In[ ]:


CODED_BOTS = sorted(['../working/' + b for b in os.listdir('../working') if '.py' in b and 'submission' not in b])

AGENTS = CODED_BOTS + SAVED_BOTS  
AGENTS = (AGENTS 
                    + [a for a in AGENTS if 'c22-' in a] 
                      + [a for a in AGENTS if 'c22-basic-raider' in a] * 8 
                          + [a for a in AGENTS if 'c30-base' in a] * 8
                              + [a for a in AGENTS if 'c31-' in a] * 1
                                 + [a for a in AGENTS if 'c33-' in a] * 3
         ) 
AGENTS = sorted(AGENTS)


# In[ ]:


printBots(AGENTS)


# In[ ]:





# ### Live Access

# In[ ]:


def show_status():
    player = observation['players'][observation['player']]
    print('   {} halite, {} bases, {} ships with {} onboard'.format( player[0], len(player[1]), 
                                            len(player[2]), sum([i[1][1] for i in player[2].items()]) )) 


# In[ ]:


LIVE = False

if LIVE:
    get_ipython().run_line_magic('run', 'submission.py')
    # reset variables
    ships_data = {};  
    max_step_time = 0 
    step_times = np.zeros(400)

    # Play as first position against random agent.
    this_run = [None, random.choice(AGENTS), random.choice(AGENTS), random.choice(AGENTS)]
    trainer = env.train(this_run)

    observation = trainer.reset()
    print(this_run)
    
    while not env.done:
        start = datetime.datetime.now()
        show_status(); print()
        
        my_action = swarm_agent(observation, env.configuration)
        step_time = (datetime.datetime.now() - start).microseconds//1e3; 
        step_times[observation.step] = step_time
       # if step_time > max_step_time: max_step_time = step_time
        print("\nStep: {}, {:.0f}ms, My Actions: {}".format(observation.step, 
                                                           step_time, 
                                                           my_action))
        observation, reward, done, info = trainer.step(my_action)
        # env.render(mode="ipython", width=100, height=90, header=False, controls=False)
    env.render()
    print(' Longest Step Time: {:.0f}ms'.format(np.max(step_times)))


# In[ ]:





# In[ ]:


if "swarm_agent" in globals():
    pd.Series(step_times, range(len(step_times))).plot()


# In[ ]:


# %load_ext line_profiler
# %lprun -f findBestSpot swarm_agent(observation, env.configuration)


# In[ ]:





# In[ ]:





# In[ ]:


if "swarm_agent" in globals():
#     print(player); print()
    for i, player in enumerate(observation['players']):
        print('{} halite, {} bases, {} ships with {} onboard'.format( player[0], len(player[1]), 
                                            len(player[2]), sum([i[1][1] for i in player[2].items()])),
                  (' <---' if observation['player'] == i else ''))


# In[ ]:





# In[ ]:





# # Evaluate your Agent

# In[ ]:


def mean_reward(rewards):
    wins = 0
    ties = 0
    loses = 0
    for r in rewards:
        r0 = 0 if r[0] is None else r[0]
        r1 = 0 if r[1] is None else r[1]
        if r0 > r1:
            wins += 1
        elif r1 > r0:
            loses += 1
        else:
            ties += 1
#     return f'wins={wins/len(rewards)}, ties={ties/len(rewards)}, loses={loses/len(rewards)}'
    return "{:.0%} wins, {:.0%} losses, {:.0%} ties".format(
        wins/len(rewards), loses/len(rewards), ties/len(rewards) )


# In[ ]:





# ##### Run Tests

# In[ ]:


def show():
    field = ["submission.py", random.choice(AGENTS), random.choice(AGENTS), random.choice(AGENTS)]
    print(field)
    env.reset()
    env.run(field)
    env.render(mode="ipython", width=800, height=600)


# In[ ]:


# show()
# show()
# show()


# In[ ]:





# In[ ]:





# In[ ]:


# ( np.array(obs.halite).reshape((21,21)) )


# In[ ]:


printBots(AGENTS)


# In[ ]:





# In[ ]:


LIVE_AGENT = SAVED_BOTS[-5] 
LIVE_AGENT = 'submission.py'
print(LIVE_AGENT)


# In[ ]:


N_RUNS = 40
CSEED = 19583


# In[ ]:





# In[ ]:


def compete(runs):
    return evaluate("halite", [ LIVE_AGENT, random.choice(AGENTS), random.choice(AGENTS), random.choice(AGENTS)],
    num_episodes=runs, configuration={"agentExec": "LOCAL"})


# In[ ]:


start = datetime.datetime.now()
if CSEED > 0:
    random.seed(CSEED)
else:
    random.seed(datetime.datetime.now().microsecond)
    
preds = Parallel(n_jobs=4)(delayed(compete)(run) for run in [(i + N_RUNS) // 4 for i in range(4)])

print('Over {} runs:'.format(N_RUNS))
print(" ", mean_reward([p for sub in preds for p in sub]),' for ', LIVE_AGENT )
print('\nTime Elapsed: ', (datetime.datetime.now() - start).seconds, 's')


# In[ ]:




