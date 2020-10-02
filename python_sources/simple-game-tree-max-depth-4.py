#!/usr/bin/env python
# coding: utf-8

# # Install kaggle-environments

# In[ ]:


# 1. Enable Internet in the Kernel (Settings side pane)

# 2. Curl cache may need purged if v0.1.4 cannot be found (uncomment if needed). 
# !curl -X PURGE https://pypi.org/simple/kaggle-environments

# ConnectX environment was defined in v0.1.4
get_ipython().system("pip install 'kaggle-environments>=0.1.4'")


# # Create ConnectX Environment

# In[ ]:


from kaggle_environments import evaluate, make

env = make("connectx", debug=True)
env.render()


# # Create an Agent
# 
# To create the submission, an agent function should be fully encapsulated (no external dependencies).  
# 
# When your agent is being evaluated against others, it will not have access to the Kaggle docker image.  Only the following can be imported: Python Standard Library Modules, gym, numpy, scipy (more may be added later). 

# In[ ]:


def my_agent(observation, configuration):
    import numpy as np
    max_depth = 4
    def check_win(bord):
        for y in range(bord.shape[0]):
            for x in range(bord.shape[1]):
                if x+3 < bord.shape[1] and bord[y,x] != 0:
                    if bord[y,x] == bord[y,x+1] and                        bord[y,x+1] == bord[y,x+2] and                        bord[y,x+2] == bord[y,x+3]:
                        return bord[y,x]
                if y+3 < bord.shape[0] and bord[y,x] != 0:
                    if bord[y,x] == bord[y+1,x] and                        bord[y+1,x] == bord[y+2,x] and                        bord[y+2,x] == bord[y+3,x]:
                        return bord[y,x]
                if x+3 < bord.shape[1] and y+3 < bord.shape[0] and bord[y,x] != 0:
                    if bord[y,x] == bord[y+1,x+1] and                        bord[y+1,x+1] == bord[y+2,x+2] and                        bord[y+2,x+2] == bord[y+3,x+3]:
                        return bord[y,x]
                if x+3 < bord.shape[1] and y+3 < bord.shape[0] and bord[y+3,x] != 0:
                    if bord[y+3,x] == bord[y+2,x+1] and                        bord[y+2,x+1] == bord[y+1,x+2] and                        bord[y+1,x+2] == bord[y,x+3]:
                        return bord[y+3,x]
        return 0
    def make_score(bord):
        """
        If the game does not advance to the conclusion even after calculating up to max_depth, 
        a score is obtained from the board. Here, the score is simply set to be advantageous 
        to the side where the stones are gathered.
        """
        d = np.where(bord == 1)
        s = -np.std(d[1])
        if d[0].shape[0] > 0:
            mx, my = np.mean(d[0]), np.mean(d[1])
            s -= np.std([np.sqrt(((x-mx)*(x-mx))+((y-my)*(y-my))) for x,y in zip(d[0],d[1])])
        d = np.where(bord == 2)
        s += np.std(d[1])
        if d[0].shape[0] > 0:
            mx, my = np.mean(d[0]), np.mean(d[1])
            s += np.std([np.sqrt(((x-mx)*(x-mx))+((y-my)*(y-my))) for x,y in zip(d[0],d[1])])
        return s
    def drop_one(c, b, stack_bord):
        nonlocal configuration
        bord = stack_bord.copy()
        d = np.where(bord[:,c] != 0)[0]
        p = bord.shape[0] if d.shape[0] == 0 else min(d)
        if p == 0:
            return None
        else:
            bord[p-1,c] = b
            return bord
    def grow_tree(selection, tree, depth, current_bord, b):
        nonlocal configuration, max_depth
        if depth >= max_depth:
            return
        w = check_win(current_bord)
        if w != 0:
            s = np.inf if b == 2 else -np.inf
        else:
            s = make_score(current_bord)
        leaf = {'root':tree,'selection':selection,'bord':current_bord,'score':s,'next':[]}
        tree['next'].append(leaf)
        if w != 0:
            return
        k = 1 if b==2 else 2
        for i in range(configuration.columns):
            t = drop_one(i, b, current_bord)
            if t is not None:
                grow_tree(i, leaf, depth+1, t, k)
    
    game_tree = {'next':[]}
    current_bord = np.array(observation['board'], dtype=np.uint8).reshape((6,7))
    if observation.mark == 2:
        # Decide that I am 1
        current_bord[current_bord==2] = 255
        current_bord[current_bord==1] = 2
        current_bord[current_bord==255] = 1
    grow_tree(0, game_tree, 0, current_bord, 1)
    game_tree = game_tree['next'][0]
    def minmax(b, tree):
        nonlocal configuration, max_depth
        if len(tree['next']) == 0:
            return tree
        if b == 2:
            return sorted(tree['next'], key=lambda x: minmax(1, x)['score'])[0]
        else:
            return sorted(tree['next'], key=lambda x: -(minmax(2, x)['score']))[0]
    r = minmax(1, game_tree)
    while r['root'] != game_tree:
        r = r['root']    
    return r['selection']


# # Test your Agent

# In[ ]:


env.reset()
# Play as the first agent against default "random" agent.
env.run([my_agent, "random"])
env.render(mode="ipython", width=500, height=450)


# # Debug/Train your Agent

# In[ ]:


# Play as first position against random agent.
trainer = env.train([None, "random"])

observation = trainer.reset()

while not env.done:
    my_action = my_agent(observation, env.configuration)
    print("My Action", my_action)
    observation, reward, done, info = trainer.step(my_action)
    # env.render(mode="ipython", width=100, height=90, header=False, controls=False)
env.render()


# # Evaluate your Agent

# In[ ]:


def mean_reward(rewards):
    return sum(r[0] for r in rewards) / sum(r[0] + r[1] for r in rewards)

# Run multiple episodes to estimate it's performance.
print("My Agent vs Random Agent:", mean_reward(evaluate("connectx", [my_agent, "random"], num_episodes=10)))
print("My Agent vs Negamax Agent:", mean_reward(evaluate("connectx", [my_agent, "negamax"], num_episodes=10)))
print("Random Agent vs My Agent:", mean_reward(evaluate("connectx", ["random", my_agent], num_episodes=10)))
print("Negamax Agent vs My Agent:", mean_reward(evaluate("connectx", ["negamax", my_agent], num_episodes=10)))


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

write_agent_to_file(my_agent, "submission.py")


# # Submit to Competition
# 
# 1. Commit this kernel.
# 2. View the commited version.
# 3. Go to "Data" section and find submission.py file.
# 4. Click "Submit to Competition"
# 5. Go to [My Submissions](https://kaggle.com/c/connectx/submissions) to view your score and episodes being played.

# In[ ]:


get_ipython().system('cat submission.py')

