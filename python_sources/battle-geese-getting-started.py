#!/usr/bin/env python
# coding: utf-8

# # Install kaggle-environments

# In[ ]:


# 1. Enable Internet in the Kernel (Settings side pane)

# 2. Curl cache may need purged if v0.1.6 cannot be found (uncomment if needed). 
# !curl -X PURGE https://pypi.org/simple/kaggle-environments

# Battle Geese environment was defined in v0.2.1
get_ipython().system("pip install 'kaggle-environments>=0.2.1'")


# # Create Battle Geese Environment

# In[ ]:


from kaggle_environments import evaluate, make

env = make("battlegeese", debug=True)
env.render()


# # Create a Submission (agent)
# 
# To submit to the competition, a python file must be created where the last function is the "act" (the function which given an observation generates an action).  Logic above the "act" function is allowed including helpers.  Any python that executes immediately will be run during the initialize phase and not included in the "act timeout".
# 
# When your agent is being evaluated against others, it will not have access to the Kaggle docker image. Only the following can be imported: Python Standard Library Modules, gym, numpy, scipy, pytorch (1.3.1, cpu only), and more may be added later.

# In[ ]:


get_ipython().run_cell_magic('writefile', 'submission.py', '\n# Silly agent which circles the perimeter clockwise.\ndef act(observation, configuration):\n    cols = configuration.columns\n    rows = configuration.rows\n    goose_head = observation.geese[observation.index][0]\n    if goose_head < cols - 1:\n        return "E"\n    elif goose_head % cols == 0:\n        return "N"\n    elif goose_head >= cols * (rows - 1):\n        return "W"\n    else:\n        return "S"')


# # Test your Agent

# In[ ]:


# Play against yourself without an ERROR or INVALID.
# Note: The first episode in the competition will run this to weed out erroneous agents.
env.run(["/kaggle/working/submission.py", "/kaggle/working/submission.py"])
print("VALID SUBMISSION!" if env.toJSON()["statuses"] == ["INACTIVE", "INACTIVE"] else "INVALID SUBMISSION!")

# Play as the first agent against default "shortest" agent.
env.run(["/kaggle/working/submission.py", "shortest"])
env.render(mode="ipython", width=800, height=600)


# # Debug/Train your Agent

# In[ ]:


# Play as first position against random agent.
trainer = env.train([None, "random"])

observation = trainer.reset()

def my_agent(observation, configuration):
    cols = configuration.columns
    rows = configuration.rows
    goose_head = observation.geese[observation.index][0]
    if goose_head < cols - 1:
        return "E"
    elif goose_head % cols == 0:
        return "N"
    elif goose_head >= cols * (rows - 1):
        return "W"
    else:
        return "S"

while not env.done:
    my_action = my_agent(observation, env.configuration)
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
    "battlegeese",
    ["/kaggle/working/submission.py", "random"],
    num_episodes=20, configuration={"agentExec": "LOCAL"}
)))
print("My Agent vs Shortest Agent:", mean_reward(evaluate(
    "battlegeese",
    ["/kaggle/working/submission.py", "shortest"],
    num_episodes=20, configuration={"agentExec": "LOCAL"}
)))


# # Play your Agent
# 
# Use the arrow keys to navigate the white goose.

# In[ ]:


# "None" represents which agent you'll manually play as.
env.play([None, "shortest", "shortest", "shortest"], width=800, height=600)


# # Submit to Competition
# 
# 1. Commit this kernel.
# 2. View the commited version.
# 3. Go to "Data" section and find submission.py file.
# 4. Click "Submit to Competition"
# 5. Go to [My Submissions](https://kaggle.com/c/battlegeese/submissions) to view your score and episodes being played.
