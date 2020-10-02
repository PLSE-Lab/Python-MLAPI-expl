# imports
import numpy as np
import pandas as pd
import random, time
from keras import Model, Sequential
from keras.models import model_from_json
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Flatten, Input
from keras.utils import to_categorical


def load_data(size,file_path):
    """
    Modified to load a random number of quizzes from the 1 million database
    """
    f=file_path
#    f='../input/sudoku/sudoku.csv'
#    size=1
    num_lines = sum(1 for l in open(f))
    skip_idx = random.sample(range(1, num_lines), num_lines - size - 1)
    sudokus = pd.read_csv(f,skiprows=skip_idx).values
    quizzes, solutions = sudokus.T

    flatX = np.array([np.reshape([int(d) for d in flatten_grid], (9, 9))
                      for flatten_grid in quizzes])
    flaty = np.array([np.reshape([int(d) for d in flatten_grid], (9, 9))
                      for flatten_grid in solutions])
    
    return (flatX, flaty)
    # return the quizzs as flatX, answer as flaty


def diff(grids_true, grids_pred):
    """
    This function shows how well predicted quizzes fit to actual solutions.
    It will store sum of differences for each pair (solution, guess)
    -------
    diff (np.array), shape (?,): Number of differences for each pair (solution, guess)
    """
    return (grids_true != grids_pred).sum((1, 2))



def batch_smart_solve(grids, solver):
    """
    NOTE : This function is ugly, feel free to optimize the code ...
    
    This function solves quizzes in the "smart" way. 
    It will fill blanks one after the other. Each time a digit is filled, 
    the new grid will be fed again to the solver to predict the next digit. 
    Again and again, until there is no more blank
    
    Parameters
    ----------
    grids (np.array), shape (?, 9, 9): Batch of quizzes to solve (smartly ;))
    solver (keras.model): The neural net solver
    
    Returns
    -------
    grids (np.array), shape (?, 9, 9): Smartly solved quizzes.
    """
    grids = grids.copy()
    Max_Zero_Num=(grids == 0).sum((1, 2)).max()
    for j in range(Max_Zero_Num):
        preds = np.array(solver.predict(to_categorical(grids)))  # get predictions
        probs = preds.max(2).T  # get highest probability for each 81 digit to predict
        values = preds.argmax(2).T + 1  # get corresponding values
        zeros = (grids == 0).reshape((grids.shape[0], 81))  # get blank positions
#        print("""Filling number {} in 81""".format(81-Max_Zero_Num+j+1))

        for grid, prob, value, zero in zip(grids, probs, values, zeros):
            if any(zero):  # don't try to fill already completed grid
                where = np.where(zero)[0]  # focus on blanks only
                confidence_position = where[prob[zero].argmax()]  # best score FOR A ZERO VALUE (confident blank)
                confidence_value = value[confidence_position]  # get corresponding value
                grid.flat[confidence_position] = confidence_value  # fill digit inplace
    return grids
    
###
#This part below is to load the trained model
#Model and weights are separated to different files for easier handling 
input_shape = (9, 9, 10)
json_file = open('../input/noprofile-nn-sudoku-test/trained_model.json','r')
model_json = json_file.read()
json_file.close()
solver = model_from_json(model_json)
solver.load_weights('../input/noprofile-nn-sudoku-test/trained_model_weights.h5')
###

runtimes=np.zeros((10,3))

for i in range(10):

    size=1
    quizzes, true_grids = load_data(size,'../input/sudoku/sudoku.csv')  # We won't use _. We will work directly with ytrain

    ### Measure how long it takes to solve one quize
    start_time=time.time()
    smart_guesses = batch_smart_solve(quizzes, solver)  # make smart guesses here!
    end_time=time.time()
    elapsed_time=end_time-start_time
    
    deltas = diff(true_grids, smart_guesses)  # get number of errors on each quizz

    if i==0:
        print('First Quiz:\n'+str(quizzes))
        print('Answer:\n'+str(true_grids))
    
    runtimes[i]=[i+1,deltas==0,elapsed_time]
    
print('List of run times:\nQuiz#       Correct?(1/0)        Time(s)')
print(runtimes)
