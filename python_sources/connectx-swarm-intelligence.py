#!/usr/bin/env python
# coding: utf-8

# # Install kaggle-environments

# In[ ]:


# 1. Enable Internet in the Kernel (Settings side pane)

# 2. Curl cache may need purged if v0.1.6 cannot be found (uncomment if needed). 
# !curl -X PURGE https://pypi.org/simple/kaggle-environments

# ConnectX environment was defined in v0.1.6
get_ipython().system("pip install 'kaggle-environments>=0.1.6'")


# # Create ConnectX Environment

# In[ ]:


from kaggle_environments import evaluate, make, utils

env = make("connectx", debug=True)
env.render()


# # Create an Agent
# 
# To create the submission, an agent function should be fully encapsulated (no external dependencies).  
# 
# When your agent is being evaluated against others, it will not have access to the Kaggle docker image.  Only the following can be imported: Python Standard Library Modules, gym, numpy, scipy, pytorch (1.3.1, cpu only), and more may be added later.
# 
# 

# In[ ]:


def swarm(obs, conf):
    def send_scout_carrier(x, y):
        """ send scout carrier to explore current cell and, if possible, cell above """
        points = send_scouts(x, y)
        # if cell above exists
        if y > 0:
            cell_above_points = send_scouts(x, y - 1)
            # cell above points have lower priority
            if points < m1 and points < (cell_above_points - 1):
                # current cell's points will be negative
                points -= cell_above_points
        return points
    
    def send_scouts(x, y):
        """ send scouts to get points from all axes of the cell """
        axes = explore_axes(x, y)
        points = combine_points(axes)
        return points
        
    def explore_axes(x, y):
        """
            find points, marks, zeros and amount of in_air cells of all axes of the cell,
            "NE" = North-East etc.
        """
        return {
            "NE -> SW": [
                explore_direction(x, lambda z : z + 1, y, lambda z : z - 1),
                explore_direction(x, lambda z : z - 1, y, lambda z : z + 1)
            ],
            "E -> W": [
                explore_direction(x, lambda z : z + 1, y, lambda z : z),
                explore_direction(x, lambda z : z - 1, y, lambda z : z)
            ],
            "SE -> NW": [
                explore_direction(x, lambda z : z + 1, y, lambda z : z + 1),
                explore_direction(x, lambda z : z - 1, y, lambda z : z - 1)
            ],
            "S -> N": [
                explore_direction(x, lambda z : z, y, lambda z : z + 1),
                explore_direction(x, lambda z : z, y, lambda z : z - 1)
            ]
        }
    
    def explore_direction(x, x_fun, y, y_fun):
        """ get points, mark, zeros and amount of in_air cells of this direction """
        # consider only opponents mark
        mark = 0
        points = 0
        zeros = 0
        in_air = 0
        for i in range(one_mark_to_win):
            x = x_fun(x)
            y = y_fun(y)
            # if board[x][y] is inside board's borders
            if y >= 0 and y < conf.rows and x >= 0 and x < conf.columns:
                # mark of the direction will be the mark of the first non-empty cell
                if mark == 0 and board[x][y] != 0:
                    mark = board[x][y]
                # if board[x][y] is empty
                if board[x][y] == 0:
                    zeros += 1
                    if (y + 1) < conf.rows and board[x][y + 1] == 0:
                        in_air += 1
                elif board[x][y] == mark:
                    points += 1
                # stop searching for marks in this direction
                else:
                    break
        return {
            "mark": mark,
            "points": points,
            "zeros": zeros,
            "in_air": in_air
        }
    
    def combine_points(axes):
        """ combine points of different axes """
        points = 0
        # loop through all axes
        for axis in axes:
            # if mark in both directions of the axis is the same
            # or mark is zero in one or both directions of the axis
            if (axes[axis][0]["mark"] == axes[axis][1]["mark"]
                    or axes[axis][0]["mark"] == 0 or axes[axis][1]["mark"] == 0):
                # combine points of the same axis
                points += evaluate_amount_of_points(
                              axes[axis][0]["points"] + axes[axis][1]["points"],
                              axes[axis][0]["zeros"] + axes[axis][1]["zeros"],
                              axes[axis][0]["in_air"] + axes[axis][1]["in_air"],
                              m1,
                              m2,
                              axes[axis][0]["mark"]
                          )
            else:
                # if marks in directions of the axis are different and none of those marks is 0
                for direction in axes[axis]:
                    points += evaluate_amount_of_points(
                                  direction["points"],
                                  direction["zeros"],
                                  direction["in_air"],
                                  m1,
                                  m2,
                                  direction["mark"]
                              )
        return points
    
    def evaluate_amount_of_points(points, zeros, in_air, m1, m2, mark):
        """ evaluate amount of points in one direction or entire axis """
        # if points + zeros in one direction or entire axis >= one_mark_to_win
        # multiply amount of points by one of the multipliers or keep amount of points as it is
        if (points + zeros) >= one_mark_to_win:
            if points >= one_mark_to_win:
                points *= m1
            elif points == two_marks_to_win:
                points = points * m2 + zeros - in_air
            else:
                points = points + zeros - in_air
        else:
            points = 0
        return points


    #################################################################################
    # one_mark_to_win points multiplier
    m1 = 100
    # two_marks_to_win points multiplier
    m2 = 10
    # define swarm's mark
    swarm_mark = obs.mark
    # define opponent's mark
    opp_mark = 2 if swarm_mark == 1 else 1
    # define one mark to victory
    one_mark_to_win = conf.inarow - 1
    # define two marks to victory
    two_marks_to_win = conf.inarow - 2
    # define board as two dimensional array
    board = []
    for column in range(conf.columns):
        board.append([])
        for row in range(conf.rows):
            board[column].append(obs.board[conf.columns * row + column])
    # define board center
    board_center = conf.columns // 2
    # start searching for the_column from board center
    x = board_center
    # shift to left/right from board center
    shift = 0
    # THE COLUMN !!!
    the_column = {
        "x": x,
        "points": float("-inf")
    }
    
    # searching for the_column
    while x >= 0 and x < conf.columns:
        # find first empty cell starting from bottom of the column
        y = conf.rows - 1
        while y >= 0 and board[x][y] != 0:
            y -= 1
        # if column is not full
        if y >= 0:
            # send scout carrier to get points
            points = send_scout_carrier(x, y)
            # evaluate which column is THE COLUMN !!!
            if points > the_column["points"]:
                the_column["x"] = x
                the_column["points"] = points
        # shift x to right or left from swarm center
        shift *= -1
        if shift >= 0:
            shift += 1
        x = board_center + shift
    
    # Swarm's final decision :)
    return the_column["x"]


# # Test your Agent

# In[ ]:


env.reset()
# Play as the first agent against "negamax" agent.
env.run([swarm, swarm])
#env.run([swarm, "negamax"])
env.render(mode="ipython", width=500, height=450)


# # Debug/Train your Agent

# In[ ]:


# Play as first position against negamax agent.
trainer = env.train([None, "negamax"])

observation = trainer.reset()

while not env.done:
    my_action = swarm(observation, env.configuration)
    print("My Action", my_action)
    observation, reward, done, info = trainer.step(my_action)
    # env.render(mode="ipython", width=100, height=90, header=False, controls=False)
env.render()


# # Evaluate your Agent

# In[ ]:


def mean_reward(rewards):
    return "{0} episodes: won {1}, lost {2}, draw {3}".format(
                                                           len(rewards),
                                                           sum(1 if r[0] > 0 else 0 for r in rewards),
                                                           sum(1 if r[1] > 0 else 0 for r in rewards),
                                                           sum(r[0] == r[1] for r in rewards)
                                                       )

# Run multiple episodes to estimate its performance.
print("Swarm vs Random Agent", mean_reward(evaluate("connectx", [swarm, "random"], num_episodes=10)))
print("Swarm vs Negamax Agent", mean_reward(evaluate("connectx", [swarm, "negamax"], num_episodes=10)))


# # Play your Agent
# Click on any column to place a checker there ("manually select action").

# In[ ]:


# "None" represents which agent you'll manually play as (first or second player).
env.play([swarm, None], width=500, height=450)
#env.play([None, swarm], width=500, height=450)


# # Write Submission File
# 
# 

# In[ ]:


import inspect
import os

def write_agent_to_file(function, file):
    with open(file, "a" if os.path.exists(file) else "w") as f:
        f.write(inspect.getsource(function))
        print(function, "written to", file)

write_agent_to_file(swarm, "submission.py")


# # Validate Submission
# Play your submission against itself.  This is the first episode the competition will run to weed out erroneous agents.
# 
# Why validate? This roughly verifies that your submission is fully encapsulated and can be run remotely.

# In[ ]:


# Note: Stdout replacement is a temporary workaround.
import sys
out = sys.stdout
submission = utils.read_file("/kaggle/working/submission.py")
agent = utils.get_last_callable(submission)
sys.stdout = out

env = make("connectx", debug=True)
env.run([agent, agent])
print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")


# # Submit to Competition
# 
# 1. Commit this kernel.
# 2. View the commited version.
# 3. Go to "Data" section and find submission.py file.
# 4. Click "Submit to Competition"
# 5. Go to [My Submissions](https://kaggle.com/c/connectx/submissions) to view your score and episodes being played.
