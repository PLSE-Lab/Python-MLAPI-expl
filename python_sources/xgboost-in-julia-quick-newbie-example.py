#!/usr/bin/env python
# coding: utf-8

# # Working with XGBoost in Julia
# ## A Quick Little Example from a Julia Newbie

# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Julia_prog_language.svg/320px-Julia_prog_language.svg.png)
# 
# I have seen a rising interest in the new programming language *Julia* for use in data science since it released its 1.0 version a few months ago in August. Obviously, Kaggle does not have support for Julia yet, but I know there are some like me who are interested in the language and its current capabillities! 
# 
# For those who may be interested in doing this competition with the language (whether seriously or whether to just try out *Julia*), here is a working example of XGBoost used in the Julia language. I managed to advance quite a bit in the leaderboard from my initial score using this code without adjusting that much about the model besides the number of rounds. 
# 
# Obviously, this code has to be run in a Julia environment to see results... but I wanted to provide it here. 

# In[ ]:


get_ipython().run_cell_magic('script', 'false', '\n# libraries used\n# --------------\nusing XGBoost\nusing DataFrames\nusing CSV \nusing StatsBase\nusing DelimitedFiles\n\n# initial setup\n@info "Set Working Directory..."\ncd(joinpath(homedir(), "Downloads/santander-customer-transaction-prediction"))\n\nfunction read_data()\n    @info "Input Data..."\n    train_df = CSV.read("train.csv")\n    test_df = CSV.read("test.csv")\n    return(train_df, test_df)\nend\n\ntrain_df, test_df = read_data()\n\nidx = [c for c in names(train_df) if c != :ID_code && c != :target]\n\n# XGBoost\n# -------\nnum_rounds = 500\n\n# define train sets \n# -----------------\ntrain_x = convert(Array{Float32}, train_df[idx])\ntrain_y = convert(Array{Int32}, train_df[:target])\n\n# define test set\n# ---------------\ntest_x = convert(Array{Float32}, test_df[idx])\n\ndtrain = DMatrix(train_x, label = train_y)\n\nboost = xgboost(dtrain, num_rounds, eta = .03, objective = "binary:logistic")\n\nprediction = XGBoost.predict(boost, test_x)\n\nprediction_rounded = Array{Int64, 1}(map(val -> round(val), prediction))\n\nsub = hcat(test_df[:ID_code], prediction)\nsub = DataFrame(sub)\n\n# clean up submission\n# -------------------\nrename!(sub, :x1 => :ID_code)\nrename!(sub, :x2 => :target)\n\nCSV.write("predictions.csv", sub)')


# I'm not all that familiar with Julia, but this was a fun little exercise in playing around with learning syntax and dealing with some of the constraints of Julia (such as the dataframe "type" not playing well with other packages just yet). You'll notice that the code is pretty similar to Python or R as far as XGBoost is concerned. You'll also notice that I didn't do any feature engineering, EDA, etc. These are all reasonably able to be accomplished in Julia, but I haven't quite learned how to do all those things yet since I'm still attached at the hip to R!
# 
# Hopefully this code helps someone out who might be looking to play around with *Julia* ! Though I'll probably be sticking to R and Python for now since there's no real reason for me to switch, *Julia* is definitely on my radar. 
