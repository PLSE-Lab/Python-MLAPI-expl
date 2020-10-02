from tensorflow import keras
import pickle

def save_history(history, filename):
  '''
  Save a keras model's training history, in a pickled format.
  
  Args:
    history (keras.callbacks.History): Training history of a keras model
    filename (str): Filename of output pickle file
  '''
  with open(filename, 'wb') as file_pi:
    pickle.dump((history.epoch, history.history), file_pi)

def load_history(filename):
  '''
  Load a keras model's training history from a pickle file.
  
  Args:
    filename (str): Filename of pickled model history data
  
  Returns:
    history (keras.callbacks.History): A keras model's training history
  '''
  history = keras.callbacks.History()
  with open(filename, 'rb') as file_pi:
    (history.epoch, history.history) = pickle.load(file_pi)
  return history

def save_earlystopping(earlystopping, filename):
  '''
  Save a model's early stopping data, in a pickled format.
  
  Args:
    earlystopping (keras.callbacks.EarlyStopping): Early stopping callback of a keras model
    filename (str): Filename of output pickle file
  '''
  with open(filename, 'wb') as file_pi:
    pickle.dump((earlystopping.stopped_epoch, earlystopping.patience,
                earlystopping.monitor, earlystopping.min_delta,
                earlystopping.monitor_op, earlystopping.restore_best_weights,
                earlystopping.wait, earlystopping.baseline), file_pi)

def load_earlystopping(filename):
  '''
  Load a model's early stopping data from a pickle file.
  
  Args:
    filename (str): Filename of pickled early stopping data

  Returns:
    earlystopping (keras.callbacks.EarlyStopping): A trained keras model's early stopping callback
  '''
  earlystopping = keras.callbacks.EarlyStopping()
  with open(filename, 'rb') as file_pi:
    (earlystopping.stopped_epoch, earlystopping.patience,
     earlystopping.monitor, earlystopping.min_delta,
     earlystopping.monitor_op, earlystopping.restore_best_weights,
     earlystopping.wait, earlystopping.baseline) = pickle.load(file_pi)
  return earlystopping

def check_earlystopping(history, earlystopping):
  '''
  Gives diagnostic information on early stopping results for a given keras model.

  Args:
    history (keras.callbacks.History): A keras model's training history
    earlystopping (keras.callbacks.EarlyStopping): A trained keras model's early stopping callback
  '''
  print('Early stopping results')
  monitor = history.history[earlystopping.monitor]
  fun1, fun2 = np.min, np.argmin
  best = 'Lowest'
  if earlystopping.monitor_op == np.greater:
    fun1, fun2 = np.max, np.argmax
    best = 'Highest'
  print(f'  Monitor: {earlystopping.monitor}')
  print(f'    {best} value: {fun1(monitor)}')
  print(f'    Epoch: {fun2(monitor)+1}')
  stopped_epoch = earlystopping.stopped_epoch - earlystopping.patience + 1
  if earlystopping.stopped_epoch == 0:
    stopped_epoch = 'None (stopped_epoch==0)'
  print(f'  Epoch detected by early stopping: {stopped_epoch}')
  if not earlystopping.restore_best_weights or earlystopping.stopped_epoch == 0:
    print('  Best weights NOT returned')
  else:
    print('  Best weights returned')
