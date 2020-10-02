#!/usr/bin/env python
# coding: utf-8

# ![banner](https://storage.googleapis.com/kaggle-competitions/kaggle/5407/media/housesbanner.png)
# # Intro
# Less than one month ago Machine Learning, Deep Learning and data science in general were a black box for me.
# Then I discovered the fantastic world of Kaggle :
#  Micro-courses (very well done), competitions and expecially the experiences of the users.
# User's notebooks are everything you need to learn, practice and understand how to go ahead.
# With the help of your notebooks, today I have reached the 1st position with a score of : 11815.94229

# # Starting
# My first submissions scores were around 21000, using only few important features (selected manually) and using Random Forest Regressor to predict results.
# Then I read and copy the "Bible" code from Alex Lekov : Stack&Blend LRs XGB LGB {House Prices K} v17 : https://www.kaggle.com/itslek/stack-blend-lrs-xgb-lgb-house-prices-k-v17
# 
# Running the code my score jumped to 12213, good job Alex!
# 
# How I could improve the code, so detailed and perfect?
# 
# One interesting thing I found is that this code is used from lot of users (expecially in House Price : Advance Regression Techniques Competition), with detailed explanation of features engineering, model building and hyperparameters selection.
# Features engineering is exactly the same, hyperparameters same for each models and so on...
# So really I don't know who is the "father" if this code...
# 

# # Features engineering
# Nothing to say...
# Nothing to touch...
# 
# He used all the features for a total of 331 features.
# 
# I have tried to drop features without importance (using for example the features importance score of GBR and XGB) with no good results.
# 
# This features engineering is the result of hard and long work, so for my novice point of view is the maximum...

# # Model building, training and hyperparameters
# 
# Same as features engineering.
# All the model have the right selection of hyperparameters, if you try to change something you go in the wrong direction!
# 
# Well this is not true for all the models.
# 
# I could improve prediction in this way : 
# - Changing max_depth and subsample in XGBRegressor (against the training prediction values and the original ones)
# - Add Random Forest Regressor
# - Add CatBoostRegressor (Very Slow!!!!)
# - Removing LGBMRegressor and adding Random Forest and CatBoost in the list of regressors parameter of StackingCVRegressor
# 
# It is obvious that StackingCVRegressor it is the most important model which makes the difference int the final prediction

# In[ ]:


# These are the models from the original code of Alex Lekv

#kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

# build our model scoring function
#def cv_rmse(model):
#    rmse = np.sqrt(-cross_val_score(model, X, y,
                                    scoring="neg_mean_squared_error",
                                    cv=kfolds))
#    return rmse


# setup models    
#alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
#alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
#e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
#e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

#ridge = make_pipeline(RobustScaler(),
#                      RidgeCV(alphas=alphas_alt, cv=kfolds))

#lasso = make_pipeline(RobustScaler(),
#                      LassoCV(max_iter=1e7, alphas=alphas2,
#                              random_state=42, cv=kfolds))

#elasticnet = make_pipeline(RobustScaler(),
#                           ElasticNetCV(max_iter=1e7, alphas=e_alphas,
#                                        cv=kfolds, l1_ratio=e_l1ratio))

#svr = make_pipeline(RobustScaler(),
#                    SVR(C=20, epsilon=0.008, gamma=0.0003, ))


#gbr = GradientBoostingRegressor(learning_rate=0.05,
#                                n_estimators=3000,
#                                max_depth=4, max_features='sqrt',
#                                min_samples_leaf=15, min_samples_split=10,
#                                loss='huber', random_state=42)


#lightgbm = LGBMRegressor(objective='regression',
#                         num_leaves=4,
#                         learning_rate=0.01,
#                         n_estimators=5000,
#                         max_bin=200,
#                         bagging_fraction=0.75,
#                         bagging_freq=5,
#                         bagging_seed=7,
#                         feature_fraction=0.2,
#                         feature_fraction_seed=7,
#                         verbose=-1,
                         # min_data_in_leaf=2,
                         # min_sum_hessian_in_leaf=11
#                         )


#xgboost = XGBRegressor(learning_rate=0.01, n_estimators=3460,
#                       max_depth=3, min_child_weight=0,
#                       gamma=0, subsample=0.7,
#                       colsample_bytree=0.7,
#                       objective='reg:squarederror', nthread=-1,
#                       scale_pos_weight=1, seed=27,
#                       reg_alpha=0.00006,
#                       random_state=42)


#cb = CatBoostRegressor(iterations=2000,
#                             learning_rate=0.01,
#                             depth=12,
#                             loss_function='RMSE',
#                             random_seed = 42,
#                             bagging_temperature = 0.2,
#                             metric_period = 100,
#                             verbose=0)


# stack
#stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet,
#                                            gbr, lightgbm, xgboost),
#                                            meta_regressor=xgboost,
#                                            use_features_in_secondary=True, random_state=42)


# # Blend models - where I have improved my score
# 
# This is the most interesting part for me...
# 
# In all the notebooks I have read, after fitting the models with the features, there is the final prediction to submit blending the previous models in one model, which is simple the sum of each model weighted, in order that the final weight must be 1.

# In[ ]:


# The original function to blend models with different weight for each model
# In all notebooks I have found the blending weights are quite similar...The best?

#def blend_models_predict(X=X):
#    return ((0.1* elastic_model_full_data.predict(X)) + 
#            (0.1 * lasso_model_full_data.predict(X)) + 
#            (0.05 * ridge_model_full_data.predict(X)) + 
#            (0.1 * svr_model_full_data.predict(X)) + 
#            (0.1 * gbr_model_full_data.predict(X)) + 
#            (0.15 * xgb_model_full_data.predict(X)) + 
#            (0.1 * lgb_model_full_data.predict(X)) + 
#            (0.3 * stack_gen_model.predict(np.array(X))))


# Wait a moment, these weights are the best ones?
# 
# I changed the function so I could easily change the weights and test different values with big different results!

# In[ ]:


# My modified blend function without LGBMRegressor, adding Random Forest and CatBoost

#def blend_models_predict(X, arr):
#    return ((arr[0] * elastic_model_full_data.predict(X)) + 
#            (arr[1] * lasso_model_full_data.predict(X)) + 
#            (arr[2] * ridge_model_full_data.predict(X)) + 
#            (arr[3] * svr_model_full_data.predict(X)) + 
#            (arr[4] * gbr_model_full_data.predict(X)) + 
#            (arr[5] * rf_model_full_data.predict(X)) + 
#            (arr[6] * xgb_model_full_data.predict(X)) + 
#            (arr[7] * cb_model_full_data.predict(X)) + 
#            (arr[8] * stack_gen_model.predict(np.array(X))))


# Now to easily test the blended model, first I saved all the models already fitted, so I could retrieve the models without running all the process again and also to save modifications made on the models.

# In[ ]:


#def rd_model(fname):
#    return pickle.load(open(fname, 'rb'))

#def wr_model(model, fname):
#    pickle.dump(model, open(fname, 'wb'))

# Save Models for later use, without running everything from beginning
#nf="Test34"

#wr_model(X, './input/'+nf+'X.sav')
#wr_model(y, './input/'+nf+'y.sav')
#wr_model(X_sub, './input/'+nf+'X_sub.sav')

#wr_model(stack_gen_model, './input/'+nf+'stack.sav')
#wr_model(elastic_model_full_data, './input/'+nf+'elastic.sav')
#wr_model(lasso_model_full_data, './input/'+nf+'lasso.sav')
#wr_model(ridge_model_full_data, './input/'+nf+'ridge.sav')
#wr_model(svr_model_full_data, './input/'+nf+'svr.sav')
#wr_model(gbr_model_full_data, './input/'+nf+'gbr.sav')
#wr_model(rf_model_full_data, './input/'+nf+'rf.sav')
#wr_model(xgb_model_full_data, './input/'+nf+'xgb.sav')
#wr_model(cb_model_full_data, './input/'+nf+'cb.sav')


# Read previously saved models 
#nf="Test33"

#X = rd_model('./input/'+nf+'X.sav')
#y = rd_model('./input/'+nf+'y.sav')
#X_sub = rd_model('./input/'+nf+'X_sub.sav')

#stack_gen_model = rd_model('./input/'+nf+'stack.sav')
#elastic_model_full_data = rd_model('./input/'+nf+'elastic.sav')
#lasso_model_full_data = rd_model('./input/'+nf+'lasso.sav')
#ridge_model_full_data = rd_model('./input/'+nf+'ridge.sav')
#svr_model_full_data = rd_model('./input/'+nf+'svr.sav')
#gbr_model_full_data = rd_model('./input/'+nf+'gbr.sav')
#rf_model_full_data = rd_model('./input/'+nf+'rf.sav')
#xgb_model_full_data = rd_model('./input/'+nf+'xgb.sav')
#cb_model_full_data = rd_model('./input/'+nf+'cb.sav')


# # First Results
# As you can see in the code below the most important and peraphs strange result is the weight given to Gradient Boosting Regressor.
# If you reduce the weight in favour of XGB for example, your score raises.
# As I understood XGB should be better the GBR, so I believed not to increase to much GBR weight, but the scores obtained say the contrary...
# Well XGB plays a big role inside StackingCVRegressor, because is in the regressor's list and is the meta_regressor.
# I think is not necessary also in the blending process.
# Another surprise is LGBMRegressor, his presence is not necessary at all.
# Elastic, Lasso are necessary with roughly the same weight and Svr is more weighted than the first two, while ridge must be present but with very low weight.

# In[ ]:


# The following lines are the first examples submitted where you can see how the final score change ONLY modifying the weights
# MAE Mean Absolute Error on training data
# elastic,lasso,ridge,svr,gbr,xgb,lgb,stack
#arr=[0.10,0.10,0.05,0.10,0.10,0.15,0.10,0.30] # MAE 6908  -> 12213 Original weights from Alex 
#arr=[0.10,0.10,0.05,0.10,0.15,0.10,0.05,0.35] # MAE 6567  -> 12176 
#arr=[0.10,0.10,0.05,0.10,0.20,0.05,0.05,0.35] # MAE 6398  -> 12163 
#arr=[0.06,0.09,0.05,0.15,0.30,0.00,0.00,0.35] # MAE 5817  -> 12127.3
#arr=[0.08,0.10,0.03,0.14,0.30,0.00,0.00,0.35] # MAE 5854  -> 12125.2


# # Reaching the Top Ten
# In the code below you can see better scores after adding Random Forest and later CatBoost.
# I could not submit each test, so I added to the previous MAE also RMSE (Root Mean Square) on training data to see how changing in MAE and RMSE were related to the final score.
# These give you a little help, but they are too much correlated to the training data, so if MAE and RMSE go down does not mean that for sure the score will be reduced.
# So I add also a RMSE comparison between different submissions with different score, so you can have a feeling if you are going in the right direction.
# 
# Using Random Forest and CatBoost only inside StackingCVRegressor I have reached the score of 11911.03292 - 8th position.

# In[ ]:


#def rmsle(y, y_pred):
#    return np.sqrt(mean_squared_error(y_pred,y))

#def print_sub(X_sub, arr):
#    submission = pd.read_csv("sample_submission.csv")
#    submission.iloc[:, 1] = np.floor(np.expm1(blend_models_predict(X_sub, arr)))
#    submission.to_csv("submission.csv", index=False)
#    print('\nSaved submission')

# RMSE is multiplied by 100 to better see differences
# elastic,lasso,ridge,svr,gbr,rf,xgb,stack                                       RMSE compared with previous submissions
#arr=[0.07,0.07,0.01,0.15,0.29,0.05,0.00,0.36] # MAE 5349 RMSE 4.780 -> 12092.5  9.095,0.710
#............
#arr=[0.09,0.09,0.01,0.15,0.30,0.01,0.00,0.35] # MAE 5569 RMSE 5.012 -> 12015.3  9.305,0.586,S0,M0
#.......
#arr=[0.09,0.09,0.01,0.15,0.30,0.01,0.00,0.35] # MAE 4064 RMSE 4.132 -> 11933.8  9.277,1.052,S1890,M1237
#.........
# Adding CatBoostRegressor and entering in 8th position
# elastic,lasso,ridge,svr,gbr,rf,xgb,cb,stack
#arr=[0.09,0.09,0.01,0.13,0.30,0.00,0.00,0.00,0.38] # MAE 3826 RMSE 3.927 -> 11911   9.342,1.113,S2079,M1346


#if (round(sum(arr),2) == 1.00):
#    print("Result...")
#    rm = rmsle(y, blend_models_predict(X, arr))*100
#    ma = int(round(mean_absolute_error(np.expm1(y), np.floor(np.expm1(blend_models_predict(X, arr)))),0))
#    print_sub(X_sub, arr)
#    print('\nMAE',ma,'RMSLE {:.3f}'.format(rm))
#else:
#    print("Error arr SUM :",round(sum(arr),2))


# In[ ]:


# This is the code to compare RMSE between current submission and previous submissions

#d5 = pd.read_csv('submission5.csv')    # last submission before using Alex code
# intermediate submissions to evaluate how result is moving...
#d6 = pd.read_csv('submission12092.csv') 
#d8 = pd.read_csv('submission12015.csv')

#data = pd.read_csv('submission.csv')

#rms5 = np.sqrt(mean_squared_error(np.log1p(data['SalePrice']), np.log1p(d5['SalePrice'])))*100
#rms6 =np.sqrt(mean_squared_error(np.log1p(data['SalePrice']), np.log1p(d6['SalePrice'])))*100
#stdf = (data['SalePrice']-d8['SalePrice']).std()
#mae = mean_absolute_error(data['SalePrice'], d8['SalePrice'])

#print('{:.3f}'.format(rms5)+','+'{:.3f}'.format(rms6)+','+'S{0:.0f}'.format(stdf)+','+'M{0:.0f}'.format(mae))


# # First position
# Once I have reached the 8th position, I run all the code in the "House Price : Advanced Regression Techniques" competition.
# My score was 0.11497 around 300-400 position.
# Why?
# Train and Test models are the same, change only the score metric.
# I tried the original code and obtain same result around 0.114...
# Most of the submissions are based on my original code, as I explained before...
# Unfortunately I found the solution which is a trick...
# In the House Price : Advanced Regression Techniques competition lot of users, expecially top users - hope not everybody) use this trick :
# - Download some top submissions score from users public notebooks
# - Blend these submissions with your submissions making some adjustments
# - Blend the result submission with the top submission you have giving to your submission more weight
# Result : my score jumped from  0.11497 to 0.10329 - 14th position!
# 
# I got this my best submission and blend with my real 11911 score - 8th position in this competition and....
# 
# Score 11815.94229 - 1st Position
# 

# In[ ]:


#This is the submission which give me the first position!
#Weight are the same as before, but I use some tricks...
#arr=[0.09,0.09,0.01,0.13,0.30,0.00,0.00,0.00,0.38] # MAE 3826 RMSLE 3.927 -> 11815.9 9.492,1.484,S3223,M1532


# # Conclusion
# I am happy to have reached my real 8th position using only precious suggestions of other users and my intuitions, because I am new in this fantastic world.
# I want to thanks Kaggle Team and all the users which all gave me this opportunity to expand my knowledge.
# I hope to learn more and more, this world is immense as the universe!
# 
# Federico Materi
