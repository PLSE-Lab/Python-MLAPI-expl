'''
Dated: Oct01-2018
Author: Mahesh Babu Mariappan (https://www.linkedin.com/in/mahesh-babu-mariappan)
Source code for simultaneously forecasting multiple time series taking into account their correlations.
Dataset used: systemresources-deeplearning-1000.csv
Columns:
index	(timesteps)
cpu	    (utilization)
ram	    (utilization)	
disk	    (utilization)	
network	    (utilization)


I am bounding the forecasts between 0 and 100 (because we are looking at forecasting resource utilization in terms of percentages)

Warning: program may consume a lot of cpu and ram.

Results:
If you have matplotlib installed, you should see a visualization of 1000 past timesteps, and 100 future timesteps with forecast values for multiple features. 
Time elapsed on an intel core i5 4-core cpu, 8gb ram: 00h:01m:18.26s


'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
from matplotlib import pyplot as plt
import numpy
import time
from os import path
import tempfile
import tensorflow as tf


_PATH = path.dirname(__file__)
_CSV_FILE = path.join(_PATH, '../input/systemresources-deeplearning-1000.csv')

def bound_forecasts_between_0_and_100(ndarray):
  '''I am bounding the forecasts between 0 and 100 (because we are looking at forecasting resource utilization in terms of percentages)'''
  return numpy.clip(ndarray, 0, 100)

def multiple_timeseries_forecast(
    csv_file_name=_CSV_FILE, export_directory=None, training_steps=500):
  '''Trains and evaluates a tensorflow model for simultaneously forecasting multiple time series.'''
  estimator = tf.contrib.timeseries.StructuralEnsembleRegressor(
      periodicities=[], num_features=4)
  reader = tf.contrib.timeseries.CSVReader(
      csv_file_name,
      skip_header_lines=1,
      column_names=((tf.contrib.timeseries.TrainEvalFeatures.TIMES,)
                    + (tf.contrib.timeseries.TrainEvalFeatures.VALUES,) * 4))
  train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(
      reader, batch_size=4, window_size=64)
  estimator.train(input_fn=train_input_fn, steps=training_steps)
  evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
  current_state = estimator.evaluate(input_fn=evaluation_input_fn, steps=1)
  values = [current_state["observed"]]
  times = [current_state[tf.contrib.timeseries.FilteringResults.TIMES]]
  if export_directory is None:
    export_directory = tempfile.mkdtemp()
  input_receiver_fn = estimator.build_raw_serving_input_receiver_fn()
  export_location = estimator.export_savedmodel(
      export_directory, input_receiver_fn)
  with tf.Graph().as_default():
    numpy.random.seed(1)  
    with tf.Session() as session:
      signatures = tf.saved_model.loader.load(
          session, [tf.saved_model.tag_constants.SERVING], export_location)
      for _ in range(100):
        current_prediction = (
            tf.contrib.timeseries.saved_model_utils.predict_continuation(
                continue_from=current_state, signatures=signatures,
                session=session, steps=1))
        next_sample = numpy.random.multivariate_normal(
            mean=numpy.squeeze(current_prediction["mean"], axis=(0, 1)),
            cov=numpy.squeeze(current_prediction["covariance"], axis=(0, 1)))
        filtering_features = {
            tf.contrib.timeseries.TrainEvalFeatures.TIMES: current_prediction[
                tf.contrib.timeseries.FilteringResults.TIMES],
            tf.contrib.timeseries.TrainEvalFeatures.VALUES: next_sample[
                None, None, :]}
        current_state = (
            tf.contrib.timeseries.saved_model_utils.filter_continuation(
                continue_from=current_state,
                session=session,
                signatures=signatures,
                features=filtering_features))
        values.append(next_sample[None, None, :])
        times.append(current_state["times"])
  past_and_future_values = numpy.squeeze(numpy.concatenate(values, axis=1), axis=0)
  past_and_future_timesteps = numpy.squeeze(numpy.concatenate(times, axis=1), axis=0)
  return past_and_future_timesteps, past_and_future_values


def main(unused_argv):
  startTime = time.time()
  past_and_future_timesteps, past_and_future_values = multiple_timeseries_forecast()
  endTime = time.time()

  print('len(past_and_future_timesteps)', len(past_and_future_timesteps))
  print('len(past_and_future_values)', len(past_and_future_values))
  #print('type(past_and_future_timesteps)', type(past_and_future_timesteps))    #<class 'numpy.ndarray'>
  #print('type(past_and_future_values)', type(past_and_future_values))      #<class 'numpy.ndarray'>
  #print('past_and_future_timesteps.shape', past_and_future_timesteps.shape)
  #print('past_and_future_values.shape', past_and_future_values.shape)

  #print('past_and_future_timesteps[995:1005]:', past_and_future_timesteps[995:1005])
  

  print('\n##### first 1000 samples #####')
  print('min value:', numpy.amin(past_and_future_values[:999]), 'at ', numpy.unravel_index(past_and_future_values[:999].argmin(), past_and_future_values[:999].shape))    #returns 
  print('max value:', numpy.amax(past_and_future_values[:999]), 'at ', numpy.unravel_index(past_and_future_values[:999].argmax(), past_and_future_values[:999].shape))    #returns 

  print('\n##### 100 future forecast samples #####')

  print('min value:', numpy.amin(past_and_future_values[999:]), 'at ', numpy.unravel_index(past_and_future_values[999:].argmin(), past_and_future_values[999:].shape))    #returns 

  print('max value:', numpy.amax(past_and_future_values[999:]), 'at ', numpy.unravel_index(past_and_future_values[999:].argmax(), past_and_future_values[999:].shape))    #returns 

  print('past_and_future_values[999:].shape:', past_and_future_values[999:].shape)

  print('\nAll future 100 forecast values:\n', past_and_future_values[999:])

  print('Now bounding forecasts between 0 and 100 since this is a system resource utilization problem.')
  
  #bound forecasts between 0 and 100
  past_and_future_values[999:] = bound_forecasts_between_0_and_100(past_and_future_values[999:])

  print('Done! If you have matplotlib installed, you should now see a visualization of 1000 past timesteps, and 100 future timesteps with forecast values for multiple features.')
  
  # Show where sampling starts on the plot
  plt.axvline(1000, linestyle="dotted")
  plt.plot(past_and_future_timesteps, past_and_future_values)
  plt.title('Simultaneous forecast of multiple time series features')
  plt.xlabel("Timesteps")
  plt.ylabel("Units")
  #handles, labels = ax.get_legend_handles_labels()
  #ax.legend(handles, labels)
  plt.show()

  #print elapsed time in hh:mm:ss format
  hours, rem = divmod(endTime-startTime, 3600)
  minutes, seconds = divmod(rem, 60)
  print("Time elapsed: {:0>2}h:{:0>2}m:{:05.2f}s".format(int(hours),int(minutes),seconds))
  print('finished running tftf-sep28-2018-multivariate-tensorflow-timeseries-forecasting.py')
  numpy.savetxt('timeseries-output.csv', past_and_future_values, delimiter=",")
  print('done writing output to timeseries-output.csv')

if __name__ == "__main__":
  print(tf.__version__)

  tf.app.run(main=main)

