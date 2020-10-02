#!/usr/bin/env python
# coding: utf-8

# # Install kaggle-environments

# In[ ]:


# 1. Enable Internet in the Kernel (Settings side pane)

# 2. Curl cache may need purged if v0.1.6 cannot be found (uncomment if needed). 
# !curl -X PURGE https://pypi.org/simple/kaggle-environments

# Halite environment was defined in v0.2.1
get_ipython().system("pip install 'kaggle-environments>=0.2.1'")


# # Create Halite Environment

# In[ ]:


from kaggle_environments import evaluate, make

env = make("halite", debug=True)
env.render()


# # Create a Submission (agent)
# 
# To submit to the competition, a python file must be created where the last function is the "act" (the function which given an observation generates an action).  Logic above the "act" function is allowed including helpers.  Any python that executes immediately will be run during the initialize phase and not included in the "act timeout".
# 
# When your agent is being evaluated against others, it will not have access to the Kaggle docker image. Only the following can be imported: Python Standard Library Modules, gym, numpy, scipy, pytorch (1.3.1, cpu only), and more may be added later.

# In[ ]:


get_ipython().run_cell_magic('writefile', 'submission.py', '\nfrom random import choice\ndef agent(obs):\n    action = {}\n    ship_id = list(obs.players[obs.player][2].keys())[0]\n    ship_action = choice(["NORTH", "SOUTH", "EAST", "WEST", None])\n    if ship_action is not None:\n        action[ship_id] = ship_action\n    return action')


# # Test your Agent

# In[ ]:


# Play against yourself without an ERROR or INVALID.
# Note: The first episode in the competition will run this to weed out erroneous agents.
env.run(["/kaggle/working/submission.py", "/kaggle/working/submission.py"])
print("EXCELLENT SUBMISSION!" if env.toJSON()["statuses"] == ["DONE", "DONE"] else "MAYBE BAD SUBMISSION?")

# Play as the first agent against default "shortest" agent.
env.run(["/kaggle/working/submission.py", "random"])
env.render(mode="ipython", width=800, height=600)


# # Debug/Train your Agent

# In[ ]:


# Play as first position against random agent.
trainer = env.train([None, "random"])

observation = trainer.reset()

from random import choice
def my_agent(obs):
    action = {}
    ship_id = list(obs.players[obs.player][2].keys())[0]
    ship_action = choice(["NORTH", "SOUTH", "EAST", "WEST", None])
    if ship_action is not None:
        action[ship_id] = ship_action
    return action

while not env.done:
    my_action = my_agent(observation)
    print("My Action", my_action)
    observation, reward, done, info = trainer.step(my_action)
    # env.render(mode="ipython", width=100, height=90, header=False, controls=False)
env.render()


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
    return f'wins={wins/len(rewards)}, ties={ties/len(rewards)}, loses={loses/len(rewards)}'

# Run multiple episodes to estimate its performance.
# Setup agentExec as LOCAL to run in memory (runs faster) without process isolation.
print("My Agent vs Random Agent:", mean_reward(evaluate(
    "halite",
    ["/kaggle/working/submission.py", "random"],
    num_episodes=10, configuration={"agentExec": "LOCAL"}
)))


# # Submit to Competition
# 
# 1. Commit this kernel.
# 2. View the commited version.
# 3. Go to "Data" section and find submission.py file.
# 4. Click "Submit to Competition"
# 5. Go to [My Submissions](https://kaggle.com/c/halite/submissions) to view your score and episodes being played.
