#!/usr/bin/env python
# coding: utf-8

# # AlphaZero baseline
# We open source an AlphaZero baseline in [PARL](https://github.com/PaddlePaddle/PARL/tree/develop/benchmark/torch/AlphaZero) repo to solve the Connect 4 game in Kaggle.
# 
# <img src="https://github.com/PaddlePaddle/PARL/blob/develop/.github/PARL-logo.png?raw=true" width="400px">

# ## Features
# Based on the code of [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) repo, we add following features in the baseline. 
# 
# 1. Fine-tune the hyper parameters of the AlphaZero algorithm to get a better score for the Connect 4 game.
# 2. Take advantage of the parallelism capacity of [PARL](https://github.com/PaddlePaddle/PARL) to support running self-play, pitting, dataset evaluating tasks in parallel.
# 3. Statistics `good moves rate` and `perfect moves rate` mentioned in [1k connect4 validation set](https://www.kaggle.com/petercnudde/scoring-connect-x-agents) of different iteration models and visualize these indicators in the tensorboard.

# ## How to run
# 
# 1. install dependencies:
# ```bash
# pip install parl==1.3.2 torch torchvision tqdm
# ```
# 
# 2. git clone https://github.com/PaddlePaddle/PARL.git
# 
# 3. cd PARL/benchmark/torch/AlphaZero
# 
# 4. start xparl cluster (in one machine or multiple machines)
# 
# ```bash
# ## You can change following `cpu_num` and `args.actor_nums` in the main.py based on the CPU number of your machine.
# 
# xparl start --port 8010 --cpu_num 25
# ```
# 
# ```bash
# ## [OPTIONAL] You can also run the following script in other machines to add more CPU resource to the xparl cluster, so you can increase the parallelism (args.actor_nums).
# 
# xparl connect --address MASTER_IP:8010 --cpu_num [CPU_NUM]
# ```
# 
# 5. download the [1k connect4 validation set](https://www.kaggle.com/petercnudde/1k-connect4-validation-set) to the current directory. (filename: `refmoves1k_kaggle`)
# 
# 6. run training script
# ```bash
# python main.py
# ```
# 
# 7. Visualize (good moves rate and perfect moves rate)
# ```bash
# tensorboard --logdir .
# ```

# ## How to submit
# To submit the well-trained model to the Kaggle, you can use our provided script to generate `submission.py`, for example:
# ```bash
# python gen_submission.py saved_model/best.pth.tar
# ```

# ## Performance
# Following are `good moves rate` and `perfect moves rate` indicators in the tensorbaord. 
# 
# <img src="https://github.com/PaddlePaddle/PARL/blob/develop/benchmark/torch/AlphaZero/.pic/good_moves.png?raw=true" width="400px"/> <img src="https://github.com/PaddlePaddle/PARL/blob/develop/benchmark/torch/AlphaZero/.pic/perfect_moves.png?raw=true" width="400px"/>
# 
# > It takes about 1 day to run 25 iterations on the machine with 25 cpus.
# 
# It can reach about **score 1391 and rank 5** on 2020/06/04.

# ## Reference
# - [suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general)
# - [Scoring connect-x agents](https://www.kaggle.com/petercnudde/scoring-connect-x-agents)

# In[ ]:


# output submission.py
with open("../input/welltrained/submission.py", "r") as f, open('submission.py', 'w') as out:
    out.write(f.read())

