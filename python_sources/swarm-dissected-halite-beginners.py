#!/usr/bin/env python
# coding: utf-8

# # Halite Beginner's Notebook: Disecting the Swarm

# This notebook was designed to provide a broad introduction for the Halite competition, aimed at beginners to both [Halite](https://www.kaggle.com/c/halite) and [Game AI](https://www.kaggle.com/learn/intro-to-game-ai-and-reinforcement-learning). If you found this notebook useful, or would like me to polish more, just give a comment.
# 
# After reading/working through this notebook, you will:
# 1. Have a general understanding of the Halite game and the main components
# 2. Basic knowledge of how the [Halite Swarm Intelligence](https://www.kaggle.com/yegorbiryukov/halite-swarm-intelligencebe) notebook (downloaded 6/19) works
# 3. Have the tools to debug, evaluate and submit an agent to the competition

# **Credit where credit is due.** The code used here is from [a really cool notebook, Halite Swarm Intelligence, by Yegor Biryukov](https://www.kaggle.com/yegorbiryukov/halite-swarm-intelligence), downloaded on 6/19. 

# # 1. First things first: What is Halite? 
# 
# [From the overview](https://www.kaggle.com/c/halite/overview), Halite by Two Sigma ("Halite") is a resource management game where you build and control a small armada of ships on the game board. Your algorithms determine the ship's movements to collect halite, and the agent with the most halite at the end of the match wins. Your algorithms control your fleet, build new ships, create shipyards, and mine the regenerating halite. This version of Halite contains rule changes as compared to previous versions.
# 
# 
# Let's take a look at the gameplay before we break down the components. This code will create a Halite game with Random players (agents). Make sure you have the internet enabled in the kernel (in the side pane.)

# In[ ]:


# Curl cache may need purged if v0.1.6 cannot be found (uncomment if needed). 
# !curl -X PURGE https://pypi.org/simple/kaggle-environments

# Halite environment was defined in v0.2.1
get_ipython().system("pip install 'kaggle-environments>=0.2.1'")


# In[ ]:


from kaggle_environments import evaluate, make

env = make("halite", debug=True)
env.render()


# In[ ]:


env.run(["random", "random", "random", "random"])
env.render(mode="ipython", width=800, height=600)


# ## Halite overview

# In the game above, we are watching four random agents compete. Around the edges are some basic statistics for each agent, and the middle contains the 21x21 playing field. Drag the slider to the very beginning of the game. Each player has 5000 Halite, no cargo (i.e., Halite on ships), one ship, and no shipyards. On the playing field, we see the initial ships, and variously sized Halite deposits. By using the left/right keys, you can step through the game to see the agents move about. [This notebook](https://www.kaggle.com/alexisbcook/getting-started-with-halite) also contains an introduction of these elements.
# 
# **The goal of the game is to collect as much Halite as possible.**

# ![Halite](https://i.imgur.com/3NENMos.png)
# 
# Halite is collected by ships, and deposited in the shipyards. The Halite must be deposited in a shipyard for it to count towards your winning total.

# ![Shipyards](https://i.imgur.com/LAc6fj8.png)
# 
# 
# Agents start the game with no shipyards. Shipyards are created by converting a ship into a shipyard (for a 500 Halite cost). The shipyards accept Halite, and can create new ships (for a 500 Halite cost). Agents can have multiple shipyards. If a ship collides with an enemy shipyard, destroying the ship, the ship's cargo, and the enemy shipyard. See the [full rules](https://www.kaggle.com/c/halite/overview/halite-rules) for other collisions, and how the game system resolves the collisions.

# ![Ships](https://i.imgur.com/eKN0kP3.png)
# 
# 
# Agents collect Halite with ships. A ship can only collect halite from its current position, and only 25% of the halite available in the cell. The Halite is added to the ship's "cargo". To count towards the final scores, the Halite needs to be deposited into one of their shipyards. Agents can have multiple ships. If two ships colide, the ship with **more** halite in its cargo is destroyed, and the other ship collects the destroyed ship's cargo

# # 2. How to play: Disecting the Swarm

# Now that we're familiar with the basic components of the game, it's time to think about what we need our agent to do. To win the game, we collect Halite with the ships, and deposit it in the shipyards. Therefore, to be able to compete, we'll have to (at minimum):
# - Tell our ships what to do
#     - create a shipyard?
#     - collect Halite?
#     - crash into another ship?
#     - move about on the playing board?
# - Tell our shipyards what to do:
#     - spawn ships?
#     - spawn more ships?
# 
# The [starter code](https://www.kaggle.com/c/halite/overview/getting-started) below gives a clue on how to start, but seemed a little sparse on the details. There is also the [Halite starter code](https://www.kaggle.com/alexisbcook/getting-started-with-halite), and the [Halite SDK Overview](https://www.kaggle.com/sam/halite-sdk-overview). Here, I started with the popular [Halite Swarm Intelligence](https://www.kaggle.com/yegorbiryukov/halite-swarm-intelligence) notebook, and broke it down piece by piece so we can watch how a robust agent is created. Note that this code was downloaded on 6/19, and will differ from the current version as the author is making improvements to his code.

# ## Release the Swarm!!

# At the highest level, for the agent to take a turn, it will:
# - get the current board
# - assign ship actions
# - assign shipyard actions

# In[ ]:


import random


# In[ ]:


def swarm_agent(observation, configuration):
    s_env = get_swarm_environment(observation, configuration)
    actions = actions_of_ships(s_env)
    actions = actions_of_shipyards(actions, s_env)
    return actions


# ### Getting the current state of the board

# Creates a dictionary (s_env) that contains the current map, and other variables the agent will need later with respect to the ships, shipyards and halite amounts.

# In[ ]:


def get_swarm_environment(observation, configuration):
    """ adapt environment for the Swarm """
    s_env = {}
    s_env["obs"] = observation
    if globals_not_defined:
        define_some_globals(configuration)
    s_env["map"] = get_map(s_env["obs"])
    s_env["my_halite"] = s_env["obs"].players[s_env["obs"].player][0]
    s_env["my_shipyards_coords"], s_env["my_ships_coords"] = get_my_units_coords_and_update_map(s_env)
    s_env["ships_keys"] = list(s_env["obs"].players[s_env["obs"].player][2].keys())
    s_env["ships_values"] = list(s_env["obs"].players[s_env["obs"].player][2].values())
    s_env["shipyards_keys"] = list(s_env["obs"].players[s_env["obs"].player][1].keys())
    return s_env


# Gets the map (playing board) as two dimensional array of objects and set amounts of halite in each cell.

# In[ ]:


def get_map(obs):
    game_map = []
    for x in range(conf.size):
        game_map.append([])
        for y in range(conf.size):
            game_map[x].append({
                # value will be ID of owner
                "shipyard": None,
                # value will be ID of owner
                "ship": None,
                # value will be amount of halite
                "ship_cargo": None,
                # amount of halite
                "halite": obs.halite[conf.size * y + x]
            })
    return game_map


#  Gets lists of coordinates of my units and update locations of ships and shipyards on the map

# In[ ]:


def get_my_units_coords_and_update_map(s_env):
    # arrays of (x, y) coords
    my_shipyards_coords = []
    my_ships_coords = []
    
    for player in range(len(s_env["obs"].players)):
        shipyards = list(s_env["obs"].players[player][1].values())
        for shipyard in shipyards:
            x = shipyard % conf.size
            y = shipyard // conf.size
            # place shipyard on the map
            s_env["map"][x][y]["shipyard"] = player
            if player == s_env["obs"].player:
                my_shipyards_coords.append((x, y))
        
        ships = list(s_env["obs"].players[player][2].values())
        for ship in ships:
            x = ship[0] % conf.size
            y = ship[0] // conf.size
            # place ship on the map
            s_env["map"][x][y]["ship"] = player
            s_env["map"][x][y]["ship_cargo"] = ship[1]
            if player == s_env["obs"].player:
                my_ships_coords.append((x, y))
    return my_shipyards_coords, my_ships_coords


# ## Assigning actions to the ships

# With the current map, actions can be assigned to the ships. This code iterates through the ships, and chooses between actions:
# - New ships are initialized and assigned a set of directions from the movement tactics
# - Ships will convert to a shipyard if they have enough Halite (and not at halite source ot it's last step)
# - If there is no shipyards, but enough Halite to spawn ships, they will convert to a shipyard
# - Move if the amount of Halite is low, or there is an enemy ship near

# In[ ]:


def actions_of_ships(s_env):
    """ actions of every ship of the Swarm """
    global movement_tactics_index
    actions = {}
    for i in range(len(s_env["my_ships_coords"])):
        x = s_env["my_ships_coords"][i][0]
        y = s_env["my_ships_coords"][i][1]

        # if this is a new ship
        if s_env["ships_keys"][i] not in ships_data:
            ships_data[s_env["ships_keys"][i]] = {
                "moves_done": 0,
                "ship_max_moves": random.randint(1, max_moves_amount),
                "directions": movement_tactics[movement_tactics_index]["directions"],
                "directions_index": 0
            }
            movement_tactics_index += 1
            if movement_tactics_index >= movement_tactics_amount:
                movement_tactics_index = 0

        # if ship has enough halite to convert to shipyard and not at halite source ot it's last step
        elif ((s_env["ships_values"][i][1] >= convert_threshold and s_env["map"][x][y]["halite"] == 0) or
                (s_env["obs"].step == (conf.episodeSteps - 2) and s_env["ships_values"][i][1] >= conf.convertCost)):
            actions[s_env["ships_keys"][i]] = "CONVERT"
            s_env["map"][x][y]["ship"] = None

        # if there is no shipyards and enough halite to spawn few ships
        elif len(s_env["shipyards_keys"]) == 0 and s_env["my_halite"] >= convert_threshold:
            s_env["my_halite"] -= conf.convertCost
            actions[s_env["ships_keys"][i]] = "CONVERT"
            s_env["map"][x][y]["ship"] = None
        
        else:
            # if this cell has low amount of halite or enemy ship is near
            if (s_env["map"][x][y]["halite"] < low_amount_of_halite or
                    enemy_ship_near(x, y, s_env["obs"].player, s_env["map"], s_env["ships_values"][i][1])):
                actions = move_ship(x, y, actions, s_env, i)
    return actions


# Get the directions for a ship to move

# In[ ]:


# list of directions
directions_list = [
    {
        "direction": "NORTH",
        "x": lambda z: z,
        "y": lambda z: get_c(z - 1)
    },
    {
        "direction": "EAST",
        "x": lambda z: get_c(z + 1),
        "y": lambda z: z
    },
    {
        "direction": "SOUTH",
        "x": lambda z: z,
        "y": lambda z: get_c(z + 1)
    },
    {
        "direction": "WEST",
        "x": lambda z: get_c(z - 1),
        "y": lambda z: z
    }
]


# Get list of directions in a certain sequence for the movement tactics

# In[ ]:


def get_directions(i0, i1, i2, i3):
    return [directions_list[i0], directions_list[i1], directions_list[i2], directions_list[i3]]


# List of the different movement tactics

# In[ ]:


movement_tactics = [
    # N -> E -> S -> W
    {"directions": get_directions(0, 1, 2, 3)},
    # S -> E -> N -> W
    {"directions": get_directions(2, 1, 0, 3)},
    # N -> W -> S -> E
    {"directions": get_directions(0, 3, 2, 1)},
    # S -> W -> N -> E
    {"directions": get_directions(2, 3, 0, 1)},
    # E -> N -> W -> S
    {"directions": get_directions(1, 0, 3, 2)},
    # W -> S -> E -> N
    {"directions": get_directions(3, 2, 1, 0)},
    # E -> S -> W -> N
    {"directions": get_directions(1, 2, 3, 0)},
    # W -> N -> E -> S
    {"directions": get_directions(3, 0, 1, 2)},
]
movement_tactics_amount = len(movement_tactics)


# Logic to move the ship according to the first acceptable tactic. Tactics include:
# - Boarding:
# - Go for Halite:
# - Unload Halite:
# - Attack Shipyard:
# - Standard Patrol:

# In[ ]:


def move_ship(x_initial, y_initial, actions, s_env, ship_index):
    ok, actions = boarding(x_initial, y_initial, s_env["ships_keys"][ship_index], actions, s_env, ship_index)
    if ok:
        return actions
    ok, actions = go_for_halite(x_initial, y_initial, s_env["ships_keys"][ship_index], actions, s_env, ship_index)
    if ok:
        return actions
    ok, actions = unload_halite(x_initial, y_initial, s_env["ships_keys"][ship_index], actions, s_env, ship_index)
    if ok:
        return actions
    ok, actions = attack_shipyard(x_initial, y_initial, s_env["ships_keys"][ship_index], actions, s_env, ship_index)
    if ok:
        return actions
    return standard_patrol(x_initial, y_initial, s_env["ships_keys"][ship_index], actions, s_env, ship_index)


# When boarding ships, the agent needs to find a ship with **more** Halite than the current ship. 

# In[ ]:


def boarding(x_initial, y_initial, ship_id, actions, s_env, ship_index):
    """ Yo Ho Ho and a Bottle of Rum!!! """
    # direction of ship with biggest prize
    biggest_prize = None
    for d in range(len(directions_list)):
        x = directions_list[d]["x"](x_initial)
        y = directions_list[d]["y"](y_initial)
        # if ship is there, has enough halite and safe for boarding
        if (s_env["map"][x][y]["ship"] != s_env["obs"].player and
                s_env["map"][x][y]["ship"] != None and
                s_env["map"][x][y]["ship_cargo"] > s_env["ships_values"][ship_index][1] and
                not enemy_ship_near(x, y, s_env["obs"].player, s_env["map"], s_env["ships_values"][ship_index][1])):
            # if current ship has more than ship with biggest prize
            if biggest_prize == None or s_env["map"][x][y]["ship_cargo"] > biggest_prize:
                biggest_prize = s_env["map"][x][y]["ship_cargo"]
                direction = directions_list[d]["direction"]
                direction_x = x
                direction_y = y
    # if ship is there, has enough halite and safe for boarding
    if biggest_prize != None:
        actions[ship_id] = direction
        s_env["map"][x_initial][y_initial]["ship"] = None
        s_env["map"][direction_x][direction_y]["ship"] = s_env["obs"].player
        return True, actions
    return False, actions


#  When going for Halite, the ship will go to safe cell with enough halite, if it is found

# In[ ]:


def go_for_halite(x_initial, y_initial, ship_id, actions, s_env, ship_index):
    # biggest amount of halite among scanned cells
    most_halite = low_amount_of_halite
    for d in range(len(directions_list)):
        x = directions_list[d]["x"](x_initial)
        y = directions_list[d]["y"](y_initial)
        # if cell is safe to move in
        if (is_clear(x, y, s_env["obs"].player, s_env["map"]) and
                not enemy_ship_near(x, y, s_env["obs"].player, s_env["map"], s_env["ships_values"][ship_index][1])):
            # if current cell has more than biggest amount of halite
            if s_env["map"][x][y]["halite"] > most_halite:
                most_halite = s_env["map"][x][y]["halite"]
                direction = directions_list[d]["direction"]
                direction_x = x
                direction_y = y
    # if cell is safe to move in and has substantial amount of halite
    if most_halite > low_amount_of_halite:
        actions[ship_id] = direction
        s_env["map"][x_initial][y_initial]["ship"] = None
        s_env["map"][direction_x][direction_y]["ship"] = s_env["obs"].player
        return True, actions
    return False, actions


# Unload ship's halite if there is any and Swarm's shipyard is near

# In[ ]:


def unload_halite(x_initial, y_initial, ship_id, actions, s_env, ship_index):
    if s_env["ships_values"][ship_index][1] > 0:
        for d in range(len(directions_list)):
            x = directions_list[d]["x"](x_initial)
            y = directions_list[d]["y"](y_initial)
            # if shipyard is there and unoccupied
            if (is_clear(x, y, s_env["obs"].player, s_env["map"]) and
                    s_env["map"][x][y]["shipyard"] == s_env["obs"].player):
                actions[ship_id] = directions_list[d]["direction"]
                s_env["map"][x_initial][y_initial]["ship"] = None
                s_env["map"][x][y]["ship"] = s_env["obs"].player
                return True, actions
    return False, actions


# Attack opponent's shipyard if ship's cargo is empty or almost empty and there is enough ships in the Swarm

# In[ ]:


def attack_shipyard(x_initial, y_initial, ship_id, actions, s_env, ship_index):

    if s_env["ships_values"][ship_index][1] < conf.convertCost and len(s_env["ships_keys"]) > 10:
        for d in range(len(directions_list)):
            x = directions_list[d]["x"](x_initial)
            y = directions_list[d]["y"](y_initial)
            # if  opponent's shipyard is there and unoccupied
            if (s_env["map"][x][y]["shipyard"] != s_env["obs"].player and
                    s_env["map"][x][y]["shipyard"] != None and
                    s_env["map"][x][y]["ship"] == None):
                actions[ship_id] = directions_list[d]["direction"]
                s_env["map"][x_initial][y_initial]["ship"] = None
                s_env["map"][x][y]["ship"] = s_env["obs"].player
                return True, actions
    return False, actions


# Ship will move in expanding circles clockwise or counterclockwise until reaching maximum radius, then radius will be minimal again

# In[ ]:


def standard_patrol(x_initial, y_initial, ship_id, actions, s_env, ship_index):

    directions = ships_data[ship_id]["directions"]
    # set index of direction
    i = ships_data[ship_id]["directions_index"]
    direction_found = False
    for j in range(len(directions)):
        x = directions[i]["x"](x_initial)
        y = directions[i]["y"](y_initial)
        # if cell is ok to move in
        if (is_clear(x, y, s_env["obs"].player, s_env["map"]) and
                not enemy_ship_near(x, y, s_env["obs"].player, s_env["map"], s_env["ships_values"][ship_index][1])):
            ships_data[ship_id]["moves_done"] += 1
            # apply changes to game_map, to avoid collisions of player's ships next turn
            s_env["map"][x_initial][y_initial]["ship"] = None
            s_env["map"][x][y]["ship"] = s_env["obs"].player
            # if it was last move in this direction
            if ships_data[ship_id]["moves_done"] >= ships_data[ship_id]["ship_max_moves"]:
                ships_data[ship_id]["moves_done"] = 0
                ships_data[ship_id]["directions_index"] += 1
                # if it is last direction in a list
                if ships_data[ship_id]["directions_index"] >= len(directions):
                    ships_data[ship_id]["directions_index"] = 0
                    ships_data[ship_id]["ship_max_moves"] += 1
                    # if ship_max_moves reached maximum radius expansion
                    if ships_data[ship_id]["ship_max_moves"] > max_moves_amount:
                        ships_data[ship_id]["ship_max_moves"] = 1
            actions[ship_id] = directions[i]["direction"]
            direction_found = True
            break
        else:
            # loop through directions
            i += 1
            if i >= len(directions):
                i = 0
    # if ship is not on shipyard and surrounded by opponent's units
    # and there is enough halite to convert
    if (not direction_found and s_env["map"][x_initial][y_initial]["shipyard"] == None and
            s_env["ships_values"][ship_index][1] >= conf.convertCost):
        actions[ship_id] = "CONVERT"
        s_env["map"][x_initial][y_initial]["ship"] = None
    return actions


# ### Ship Helper Functions

# These helper functions are used in many places when moving the ships. They check if the cells are safe, or an enemy ship is near. 

# Checks to see if the cell is clear/movable. Could be an unocupied space, or a shipyard without a ship.

# In[ ]:


def is_clear(x, y, player, game_map):
    """ check if cell is safe to move in """
    # if there is no shipyard, or there is player's shipyard
    # and there is no ship
    if ((game_map[x][y]["shipyard"] == player or game_map[x][y]["shipyard"] == None) and
            game_map[x][y]["ship"] == None):
        return True
    return False


# Checks to see if an enemy ship is in one move away on the map, and has **less** Halite to determine if the ship needs to move away. 

# In[ ]:


def enemy_ship_near(x, y, player, m, cargo):
    """ check if enemy ship is in one move away from game_map[x][y] and has less halite """
    # m = game map
    n = get_c(y - 1)
    e = get_c(x + 1)
    s = get_c(y + 1)
    w = get_c(x - 1)
    if (
            (m[x][n]["ship"] != player and m[x][n]["ship"] != None and m[x][n]["ship_cargo"] < cargo) or
            (m[x][s]["ship"] != player and m[x][s]["ship"] != None and m[x][s]["ship_cargo"] < cargo) or
            (m[e][y]["ship"] != player and m[e][y]["ship"] != None and m[e][y]["ship_cargo"] < cargo) or
            (m[w][y]["ship"] != player and m[w][y]["ship"] != None and m[w][y]["ship_cargo"] < cargo)
        ):
        return True
    return False


# If a ship goes off the side of the map, it will appear on the other side. Therefore, it is a doughut-shape (not a sphere).

# In[ ]:


def get_c(c):
    """ get coordinate, considering donut type of the map """
    return c % conf.size


# ## Actions of Shipyard

# In contrast to the ships, there's not much to decide for the shipyards. This is a "swarm" strategy, so the agent will spawn as many ships as possible. 

# In[ ]:


def actions_of_shipyards(actions, s_env):
    """ actions of every shipyard of the Swarm """
    ships_amount = len(s_env["ships_keys"])
    # spawn ships from every shipyard, if possible
    # iterate through shipyards starting from last created
    for i in range(len(s_env["my_shipyards_coords"]))[::-1]:
        if s_env["my_halite"] >= conf.spawnCost and ships_amount <= spawn_limit:
            x = s_env["my_shipyards_coords"][i][0]
            y = s_env["my_shipyards_coords"][i][1]
            # if there is currently no ship on shipyard
            if is_clear(x, y, s_env["obs"].player, s_env["map"]):
                s_env["my_halite"] -= conf.spawnCost
                actions[s_env["shipyards_keys"][i]] = "SPAWN"
                s_env["map"][x][y]["ship"] = s_env["obs"].player
                ships_amount += 1
        else:
            break
    return actions


# ## Global Variables

# In[ ]:


def define_some_globals(configuration):
    """ define some of the global variables """
    global conf
    global convert_threshold
    global max_moves_amount
    global globals_not_defined
    conf = configuration
    convert_threshold = conf.convertCost + conf.spawnCost * 2
    max_moves_amount = conf.size
    globals_not_defined = False


# In[ ]:


conf = None

# max amount of moves in one direction before turning
max_moves_amount = None

# threshold of harvested by a ship halite to convert
convert_threshold = None

# object with ship ids and their data
ships_data = {}

# initial movement_tactics index
movement_tactics_index = 0

# amount of halite, that is considered to be low
low_amount_of_halite = 50

# limit of ships to spawn
spawn_limit = 50

# not all global variables are defined
globals_not_defined = True


# # Debugging your Agent

# Now that the agent is created, how can we tell what it is doing? The code below steps through the game, and prints out all of the actions at each timestep.
# - the `env.render` will print a picture at each timestep, but be aware this will create hundreds of images in your notebook

# In[ ]:


if "swarm_agent" in globals():
    # reset variables
    ships_data = {}
    ship_spawn_turn = 0
    movement_tactics_index = 0

    # Play as first position against random agent.
    trainer = env.train([None, "random","random","random"])

    observation = trainer.reset()

    while not env.done:
        my_action = swarm_agent(observation, env.configuration)
        print("Step: {0}, My Action: {1}".format(observation.step, my_action))
        observation, reward, done, info = trainer.step(my_action)
        # env.render(mode="ipython", width=100, height=90, header=False, controls=False)


# The following will allow you to watch the game. Note that there are always two more steps in the visual representation. 

# In[ ]:


env.render(mode="ipython", width=600, height=400, header=False, controls=True)


# # Evaluate your Agent

# Use this code to see how often, on average, the agent wins against the "random" agents.

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
    return f'wins={wins/len(rewards)}, ties={ties/len(rewards)}, loses={loses/len(rewards)}'

# Run multiple episodes to estimate its performance.
# Setup agentExec as LOCAL to run in memory (runs faster) without process isolation.
print("Swarm Agent vs Random Agent:", mean_reward(evaluate(
    "halite",
    [None, "random", "random", "random"],
    num_episodes=10, configuration={"agentExec": "LOCAL"}
)))


# # Test your Agent

# In[ ]:


#env.run(["submission.py", "submission.py", "submission.py", "submission.py"])
env.run([swarm_agent, "random", "random", "random"])
env.render(mode="ipython", width=800, height=600)


# # Submit the Swarm

# To make a submission, you need to have all your code in one cell, and write it to the file named "submission.py". This is the code of the original swarm all in one cell for you to submit if you wish. Note that if you made changes to the code above, it will NOT be reflected here (unless you copy/paste). This code is from 6/19, and may not reflect current versions of the original kernel.

# In[ ]:


get_ipython().run_cell_magic('writefile', 'submission.py', '# for Debug/Train previous line (%%writefile submission.py) should be commented out, uncomment to write submission.py\n\nimport random\n\n#FUNCTIONS###################################################\ndef get_map(obs):\n    """ get map as two dimensional array of objects and set amounts of halite in each cell """\n    game_map = []\n    for x in range(conf.size):\n        game_map.append([])\n        for y in range(conf.size):\n            game_map[x].append({\n                # value will be ID of owner\n                "shipyard": None,\n                # value will be ID of owner\n                "ship": None,\n                # value will be amount of halite\n                "ship_cargo": None,\n                # amount of halite\n                "halite": obs.halite[conf.size * y + x]\n            })\n    return game_map\n\ndef get_my_units_coords_and_update_map(s_env):\n    """ get lists of coords of my units and update locations of ships and shipyards on the map """\n    # arrays of (x, y) coords\n    my_shipyards_coords = []\n    my_ships_coords = []\n    \n    for player in range(len(s_env["obs"].players)):\n        shipyards = list(s_env["obs"].players[player][1].values())\n        for shipyard in shipyards:\n            x = shipyard % conf.size\n            y = shipyard // conf.size\n            # place shipyard on the map\n            s_env["map"][x][y]["shipyard"] = player\n            if player == s_env["obs"].player:\n                my_shipyards_coords.append((x, y))\n        \n        ships = list(s_env["obs"].players[player][2].values())\n        for ship in ships:\n            x = ship[0] % conf.size\n            y = ship[0] // conf.size\n            # place ship on the map\n            s_env["map"][x][y]["ship"] = player\n            s_env["map"][x][y]["ship_cargo"] = ship[1]\n            if player == s_env["obs"].player:\n                my_ships_coords.append((x, y))\n    return my_shipyards_coords, my_ships_coords\n\ndef get_c(c):\n    """ get coordinate, considering donut type of the map """\n    return c % conf.size\n\ndef clear(x, y, player, game_map):\n    """ check if cell is safe to move in """\n    # if there is no shipyard, or there is player\'s shipyard\n    # and there is no ship\n    if ((game_map[x][y]["shipyard"] == player or game_map[x][y]["shipyard"] == None) and\n            game_map[x][y]["ship"] == None):\n        return True\n    return False\n\ndef move_ship(x_initial, y_initial, actions, s_env, ship_index):\n    """ move the ship according to first acceptable tactic """\n    ok, actions = boarding(x_initial, y_initial, s_env["ships_keys"][ship_index], actions, s_env, ship_index)\n    if ok:\n        return actions\n    ok, actions = go_for_halite(x_initial, y_initial, s_env["ships_keys"][ship_index], actions, s_env, ship_index)\n    if ok:\n        return actions\n    ok, actions = unload_halite(x_initial, y_initial, s_env["ships_keys"][ship_index], actions, s_env, ship_index)\n    if ok:\n        return actions\n    ok, actions = attack_shipyard(x_initial, y_initial, s_env["ships_keys"][ship_index], actions, s_env, ship_index)\n    if ok:\n        return actions\n    return standard_patrol(x_initial, y_initial, s_env["ships_keys"][ship_index], actions, s_env, ship_index)\n\ndef boarding(x_initial, y_initial, ship_id, actions, s_env, ship_index):\n    """ Yo Ho Ho and a Bottle of Rum!!! """\n    # direction of ship with biggest prize\n    biggest_prize = None\n    for d in range(len(directions_list)):\n        x = directions_list[d]["x"](x_initial)\n        y = directions_list[d]["y"](y_initial)\n        # if ship is there, has enough halite and safe for boarding\n        if (s_env["map"][x][y]["ship"] != s_env["obs"].player and\n                s_env["map"][x][y]["ship"] != None and\n                s_env["map"][x][y]["ship_cargo"] > s_env["ships_values"][ship_index][1] and\n                not enemy_ship_near(x, y, s_env["obs"].player, s_env["map"], s_env["ships_values"][ship_index][1])):\n            # if current ship has more than ship with biggest prize\n            if biggest_prize == None or s_env["map"][x][y]["ship_cargo"] > biggest_prize:\n                biggest_prize = s_env["map"][x][y]["ship_cargo"]\n                direction = directions_list[d]["direction"]\n                direction_x = x\n                direction_y = y\n    # if ship is there, has enough halite and safe for boarding\n    if biggest_prize != None:\n        actions[ship_id] = direction\n        s_env["map"][x_initial][y_initial]["ship"] = None\n        s_env["map"][direction_x][direction_y]["ship"] = s_env["obs"].player\n        return True, actions\n    return False, actions\n    \ndef go_for_halite(x_initial, y_initial, ship_id, actions, s_env, ship_index):\n    """ ship will go to safe cell with enough halite, if it is found """\n    # biggest amount of halite among scanned cells\n    most_halite = low_amount_of_halite\n    for d in range(len(directions_list)):\n        x = directions_list[d]["x"](x_initial)\n        y = directions_list[d]["y"](y_initial)\n        # if cell is safe to move in\n        if (clear(x, y, s_env["obs"].player, s_env["map"]) and\n                not enemy_ship_near(x, y, s_env["obs"].player, s_env["map"], s_env["ships_values"][ship_index][1])):\n            # if current cell has more than biggest amount of halite\n            if s_env["map"][x][y]["halite"] > most_halite:\n                most_halite = s_env["map"][x][y]["halite"]\n                direction = directions_list[d]["direction"]\n                direction_x = x\n                direction_y = y\n    # if cell is safe to move in and has substantial amount of halite\n    if most_halite > low_amount_of_halite:\n        actions[ship_id] = direction\n        s_env["map"][x_initial][y_initial]["ship"] = None\n        s_env["map"][direction_x][direction_y]["ship"] = s_env["obs"].player\n        return True, actions\n    return False, actions\n\ndef unload_halite(x_initial, y_initial, ship_id, actions, s_env, ship_index):\n    """ unload ship\'s halite if there is any and Swarm\'s shipyard is near """\n    if s_env["ships_values"][ship_index][1] > 0:\n        for d in range(len(directions_list)):\n            x = directions_list[d]["x"](x_initial)\n            y = directions_list[d]["y"](y_initial)\n            # if shipyard is there and unoccupied\n            if (clear(x, y, s_env["obs"].player, s_env["map"]) and\n                    s_env["map"][x][y]["shipyard"] == s_env["obs"].player):\n                actions[ship_id] = directions_list[d]["direction"]\n                s_env["map"][x_initial][y_initial]["ship"] = None\n                s_env["map"][x][y]["ship"] = s_env["obs"].player\n                return True, actions\n    return False, actions\n\ndef attack_shipyard(x_initial, y_initial, ship_id, actions, s_env, ship_index):\n    """ \n        attack opponent\'s shipyard if ship\'s cargo is empty or almost empty\n        and there is enough ships in the Swarm\n    """\n    if s_env["ships_values"][ship_index][1] < conf.convertCost and len(s_env["ships_keys"]) > 10:\n        for d in range(len(directions_list)):\n            x = directions_list[d]["x"](x_initial)\n            y = directions_list[d]["y"](y_initial)\n            # if  opponent\'s shipyard is there and unoccupied\n            if (s_env["map"][x][y]["shipyard"] != s_env["obs"].player and\n                    s_env["map"][x][y]["shipyard"] != None and\n                    s_env["map"][x][y]["ship"] == None):\n                actions[ship_id] = directions_list[d]["direction"]\n                s_env["map"][x_initial][y_initial]["ship"] = None\n                s_env["map"][x][y]["ship"] = s_env["obs"].player\n                return True, actions\n    return False, actions\n    \ndef standard_patrol(x_initial, y_initial, ship_id, actions, s_env, ship_index):\n    """ \n        ship will move in expanding circles clockwise or counterclockwise\n        until reaching maximum radius, then radius will be minimal again\n    """\n    directions = ships_data[ship_id]["directions"]\n    # set index of direction\n    i = ships_data[ship_id]["directions_index"]\n    direction_found = False\n    for j in range(len(directions)):\n        x = directions[i]["x"](x_initial)\n        y = directions[i]["y"](y_initial)\n        # if cell is ok to move in\n        if (clear(x, y, s_env["obs"].player, s_env["map"]) and\n                not enemy_ship_near(x, y, s_env["obs"].player, s_env["map"], s_env["ships_values"][ship_index][1])):\n            ships_data[ship_id]["moves_done"] += 1\n            # apply changes to game_map, to avoid collisions of player\'s ships next turn\n            s_env["map"][x_initial][y_initial]["ship"] = None\n            s_env["map"][x][y]["ship"] = s_env["obs"].player\n            # if it was last move in this direction\n            if ships_data[ship_id]["moves_done"] >= ships_data[ship_id]["ship_max_moves"]:\n                ships_data[ship_id]["moves_done"] = 0\n                ships_data[ship_id]["directions_index"] += 1\n                # if it is last direction in a list\n                if ships_data[ship_id]["directions_index"] >= len(directions):\n                    ships_data[ship_id]["directions_index"] = 0\n                    ships_data[ship_id]["ship_max_moves"] += 1\n                    # if ship_max_moves reached maximum radius expansion\n                    if ships_data[ship_id]["ship_max_moves"] > max_moves_amount:\n                        ships_data[ship_id]["ship_max_moves"] = 1\n            actions[ship_id] = directions[i]["direction"]\n            direction_found = True\n            break\n        else:\n            # loop through directions\n            i += 1\n            if i >= len(directions):\n                i = 0\n    # if ship is not on shipyard and surrounded by opponent\'s units\n    # and there is enough halite to convert\n    if (not direction_found and s_env["map"][x_initial][y_initial]["shipyard"] == None and\n            s_env["ships_values"][ship_index][1] >= conf.convertCost):\n        actions[ship_id] = "CONVERT"\n        s_env["map"][x_initial][y_initial]["ship"] = None\n    return actions\n\ndef get_directions(i0, i1, i2, i3):\n    """ get list of directions in a certain sequence """\n    return [directions_list[i0], directions_list[i1], directions_list[i2], directions_list[i3]]\n\ndef enemy_ship_near(x, y, player, m, cargo):\n    """ check if enemy ship is in one move away from game_map[x][y] and has less halite """\n    # m = game map\n    n = get_c(y - 1)\n    e = get_c(x + 1)\n    s = get_c(y + 1)\n    w = get_c(x - 1)\n    if (\n            (m[x][n]["ship"] != player and m[x][n]["ship"] != None and m[x][n]["ship_cargo"] < cargo) or\n            (m[x][s]["ship"] != player and m[x][s]["ship"] != None and m[x][s]["ship_cargo"] < cargo) or\n            (m[e][y]["ship"] != player and m[e][y]["ship"] != None and m[e][y]["ship_cargo"] < cargo) or\n            (m[w][y]["ship"] != player and m[w][y]["ship"] != None and m[w][y]["ship_cargo"] < cargo)\n        ):\n        return True\n    return False\n\ndef define_some_globals(configuration):\n    """ define some of the global variables """\n    global conf\n    global convert_threshold\n    global max_moves_amount\n    global globals_not_defined\n    conf = configuration\n    convert_threshold = conf.convertCost + conf.spawnCost * 2\n    max_moves_amount = conf.size\n    globals_not_defined = False\n\ndef adapt_environment(observation, configuration):\n    """ adapt environment for the Swarm """\n    s_env = {}\n    s_env["obs"] = observation\n    if globals_not_defined:\n        define_some_globals(configuration)\n    s_env["map"] = get_map(s_env["obs"])\n    s_env["my_halite"] = s_env["obs"].players[s_env["obs"].player][0]\n    s_env["my_shipyards_coords"], s_env["my_ships_coords"] = get_my_units_coords_and_update_map(s_env)\n    s_env["ships_keys"] = list(s_env["obs"].players[s_env["obs"].player][2].keys())\n    s_env["ships_values"] = list(s_env["obs"].players[s_env["obs"].player][2].values())\n    s_env["shipyards_keys"] = list(s_env["obs"].players[s_env["obs"].player][1].keys())\n    return s_env\n    \ndef actions_of_ships(s_env):\n    """ actions of every ship of the Swarm """\n    global movement_tactics_index\n    actions = {}\n    for i in range(len(s_env["my_ships_coords"])):\n        x = s_env["my_ships_coords"][i][0]\n        y = s_env["my_ships_coords"][i][1]\n        # if this is a new ship\n        if s_env["ships_keys"][i] not in ships_data:\n            ships_data[s_env["ships_keys"][i]] = {\n                "moves_done": 0,\n                "ship_max_moves": random.randint(1, max_moves_amount),\n                "directions": movement_tactics[movement_tactics_index]["directions"],\n                "directions_index": 0\n            }\n            movement_tactics_index += 1\n            if movement_tactics_index >= movement_tactics_amount:\n                movement_tactics_index = 0\n        # if ship has enough halite to convert to shipyard and not at halite source ot it\'s last step\n        elif ((s_env["ships_values"][i][1] >= convert_threshold and s_env["map"][x][y]["halite"] == 0) or\n                (s_env["obs"].step == (conf.episodeSteps - 2) and s_env["ships_values"][i][1] >= conf.convertCost)):\n            actions[s_env["ships_keys"][i]] = "CONVERT"\n            s_env["map"][x][y]["ship"] = None\n        # if there is no shipyards and enough halite to spawn few ships\n        elif len(s_env["shipyards_keys"]) == 0 and s_env["my_halite"] >= convert_threshold:\n            s_env["my_halite"] -= conf.convertCost\n            actions[s_env["ships_keys"][i]] = "CONVERT"\n            s_env["map"][x][y]["ship"] = None\n        else:\n            # if this cell has low amount of halite or enemy ship is near\n            if (s_env["map"][x][y]["halite"] < low_amount_of_halite or\n                    enemy_ship_near(x, y, s_env["obs"].player, s_env["map"], s_env["ships_values"][i][1])):\n                actions = move_ship(x, y, actions, s_env, i)\n    return actions\n     \ndef actions_of_shipyards(actions, s_env):\n    """ actions of every shipyard of the Swarm """\n    ships_amount = len(s_env["ships_keys"])\n    # spawn ships from every shipyard, if possible\n    # iterate through shipyards starting from last created\n    for i in range(len(s_env["my_shipyards_coords"]))[::-1]:\n        if s_env["my_halite"] >= conf.spawnCost and ships_amount <= spawn_limit:\n            x = s_env["my_shipyards_coords"][i][0]\n            y = s_env["my_shipyards_coords"][i][1]\n            # if there is currently no ship on shipyard\n            if clear(x, y, s_env["obs"].player, s_env["map"]):\n                s_env["my_halite"] -= conf.spawnCost\n                actions[s_env["shipyards_keys"][i]] = "SPAWN"\n                s_env["map"][x][y]["ship"] = s_env["obs"].player\n                ships_amount += 1\n        else:\n            break\n    return actions\n\n\n#GLOBAL_VARIABLES#############################################\nconf = None\n# max amount of moves in one direction before turning\nmax_moves_amount = None\n# threshold of harvested by a ship halite to convert\nconvert_threshold = None\n# object with ship ids and their data\nships_data = {}\n# initial movement_tactics index\nmovement_tactics_index = 0\n# amount of halite, that is considered to be low\nlow_amount_of_halite = 50\n# limit of ships to spawn\nspawn_limit = 50\n# not all global variables are defined\nglobals_not_defined = True\n\n# list of directions\ndirections_list = [\n    {\n        "direction": "NORTH",\n        "x": lambda z: z,\n        "y": lambda z: get_c(z - 1)\n    },\n    {\n        "direction": "EAST",\n        "x": lambda z: get_c(z + 1),\n        "y": lambda z: z\n    },\n    {\n        "direction": "SOUTH",\n        "x": lambda z: z,\n        "y": lambda z: get_c(z + 1)\n    },\n    {\n        "direction": "WEST",\n        "x": lambda z: get_c(z - 1),\n        "y": lambda z: z\n    }\n]\n\n# list of movement tactics\nmovement_tactics = [\n    # N -> E -> S -> W\n    {"directions": get_directions(0, 1, 2, 3)},\n    # S -> E -> N -> W\n    {"directions": get_directions(2, 1, 0, 3)},\n    # N -> W -> S -> E\n    {"directions": get_directions(0, 3, 2, 1)},\n    # S -> W -> N -> E\n    {"directions": get_directions(2, 3, 0, 1)},\n    # E -> N -> W -> S\n    {"directions": get_directions(1, 0, 3, 2)},\n    # W -> S -> E -> N\n    {"directions": get_directions(3, 2, 1, 0)},\n    # E -> S -> W -> N\n    {"directions": get_directions(1, 2, 3, 0)},\n    # W -> N -> E -> S\n    {"directions": get_directions(3, 0, 1, 2)},\n]\nmovement_tactics_amount = len(movement_tactics)\n\n\n#THE_SWARM####################################################\ndef swarm_agent(observation, configuration):\n    """ RELEASE THE SWARM!!! """\n    s_env = adapt_environment(observation, configuration)\n    actions = actions_of_ships(s_env)\n    actions = actions_of_shipyards(actions, s_env)\n    return actions')


# How to Submit to the Competition
# 
# 1. Commit this kernel.
# 2. View the commited version.
# 3. Go to "Data" section and find submission.py file.
# 4. Click "Submit to Competition"
# 5. Go to [My Submissions](https://kaggle.com/c/halite/submissions) to view your score and episodes being played.

# The end! Please let me know if you'd like to see more or any other explanations! 

# In[ ]:




