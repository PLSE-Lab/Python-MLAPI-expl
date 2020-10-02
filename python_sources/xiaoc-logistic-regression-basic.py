from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

import pandas as pd
from six.moves import urllib
import tensorflow as tf


CSV_COLUMNS = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", 
    "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32",
     "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48",
      "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63",
      "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", 
      "79", "80", "81", "82", "83", "84", "85", "86", "87", "convert"
]

# Continuous base columns.
col_0 = tf.feature_column.numeric_column("0")
col_1 = tf.feature_column.numeric_column("1")
col_2 = tf.feature_column.numeric_column("2")
col_3 = tf.feature_column.numeric_column("3")
col_4 = tf.feature_column.numeric_column("4")
col_5 = tf.feature_column.numeric_column("5")
col_6 = tf.feature_column.numeric_column("6")
col_7 = tf.feature_column.numeric_column("7")
col_8 = tf.feature_column.numeric_column("8")
col_9 = tf.feature_column.numeric_column("9")
col_10 = tf.feature_column.numeric_column("10")
col_11 = tf.feature_column.numeric_column("11")
col_12 = tf.feature_column.numeric_column("12")
col_13 = tf.feature_column.numeric_column("13")
col_14 = tf.feature_column.numeric_column("14")
col_15 = tf.feature_column.numeric_column("15")
col_16 = tf.feature_column.numeric_column("16")
col_17 = tf.feature_column.numeric_column("17")
col_18 = tf.feature_column.numeric_column("18")
col_19 = tf.feature_column.numeric_column("19")
col_20 = tf.feature_column.numeric_column("20")
col_21 = tf.feature_column.numeric_column("21")
col_22 = tf.feature_column.numeric_column("22")
col_23 = tf.feature_column.numeric_column("23")
col_24 = tf.feature_column.numeric_column("24")
col_25 = tf.feature_column.numeric_column("25")
col_26 = tf.feature_column.numeric_column("26")
col_27 = tf.feature_column.numeric_column("27")
col_28 = tf.feature_column.numeric_column("28")
col_29 = tf.feature_column.numeric_column("29")
col_30 = tf.feature_column.numeric_column("30")
col_31 = tf.feature_column.numeric_column("31")
col_32 = tf.feature_column.numeric_column("32")
col_33 = tf.feature_column.numeric_column("33")
col_34 = tf.feature_column.numeric_column("34")
col_35 = tf.feature_column.numeric_column("35")
col_36 = tf.feature_column.numeric_column("36")
col_37 = tf.feature_column.numeric_column("37")
col_38 = tf.feature_column.numeric_column("38")
col_39 = tf.feature_column.numeric_column("39")
col_40 = tf.feature_column.numeric_column("40")
col_41 = tf.feature_column.numeric_column("41")
col_42 = tf.feature_column.numeric_column("42")
col_43 = tf.feature_column.numeric_column("43")
col_44 = tf.feature_column.numeric_column("44")
col_45 = tf.feature_column.numeric_column("45")
col_46 = tf.feature_column.numeric_column("46")
col_47 = tf.feature_column.numeric_column("47")
col_48 = tf.feature_column.numeric_column("48")
col_49 = tf.feature_column.numeric_column("49")
col_50 = tf.feature_column.numeric_column("50")
col_51 = tf.feature_column.numeric_column("51")
col_52 = tf.feature_column.numeric_column("52")
col_53 = tf.feature_column.numeric_column("53")
col_54 = tf.feature_column.numeric_column("54")
col_55 = tf.feature_column.numeric_column("55")
col_56 = tf.feature_column.numeric_column("56")
col_57 = tf.feature_column.numeric_column("57")
col_58 = tf.feature_column.numeric_column("58")
col_59 = tf.feature_column.numeric_column("59")
col_60 = tf.feature_column.numeric_column("60")
col_61 = tf.feature_column.numeric_column("61")
col_62 = tf.feature_column.numeric_column("62")
col_63 = tf.feature_column.numeric_column("63")
col_64 = tf.feature_column.numeric_column("64")
col_65 = tf.feature_column.numeric_column("65")
col_66 = tf.feature_column.numeric_column("66")
col_67 = tf.feature_column.numeric_column("67")
col_68 = tf.feature_column.numeric_column("68")
col_69 = tf.feature_column.numeric_column("69")
col_70 = tf.feature_column.numeric_column("70")
col_71 = tf.feature_column.numeric_column("71")
col_72 = tf.feature_column.numeric_column("72")
col_73 = tf.feature_column.numeric_column("73")
col_74 = tf.feature_column.numeric_column("74")
col_75 = tf.feature_column.numeric_column("75")
col_76 = tf.feature_column.numeric_column("76")
col_77 = tf.feature_column.numeric_column("77")
col_78 = tf.feature_column.numeric_column("78")
col_79 = tf.feature_column.numeric_column("79")
col_80 = tf.feature_column.numeric_column("80")
col_81 = tf.feature_column.numeric_column("81")
col_82 = tf.feature_column.numeric_column("82")
col_83 = tf.feature_column.numeric_column("83")
col_84 = tf.feature_column.numeric_column("84")
col_85 = tf.feature_column.numeric_column("85")
col_86 = tf.feature_column.numeric_column("86")
col_87 = tf.feature_column.numeric_column("87")

# Wide columns and deep columns.
base_columns = [
    col_0, col_1, col_2, col_3, col_4, col_5, col_6, col_7, col_8, col_9, col_10, col_11, col_12, 
    col_13, col_14, col_15, col_16, col_17, col_18, col_19, col_20, col_21, col_22, col_23, col_24,
    col_25, col_26, col_27, col_28, col_29, col_30, col_31, col_32, col_33, col_34, col_35, col_36,
    col_37, col_38, col_39, col_40, col_41, col_42, col_43, col_44, col_45, col_46, col_47, col_48,
    col_49, col_50, col_51, col_52, col_53, col_54, col_55, col_56, col_57, col_58, col_59, col_60,
    col_61, col_62, col_63, col_64, col_65, col_66, col_67, col_68, col_69, col_70, col_71, col_72,
    col_73, col_74, col_75, col_76, col_77, col_78, col_79, col_80, col_81, col_82, col_83, col_84,
    col_85, col_86, col_87,
]


def input_fn(data_file, num_epochs, shuffle):
  """Input builder function."""
  df_data = pd.read_csv(
      tf.gfile.Open(data_file),
      names=CSV_COLUMNS,
      skipinitialspace=True,
      engine="python",
      skiprows=1)
  # remove NaN elements
  df_data = df_data.dropna(how="any", axis=0)
  labels = df_data["convert"].apply(lambda x: x == 1).astype(int)
  return tf.estimator.inputs.pandas_input_fn(
      x = df_data,
      y = labels,
      batch_size = 100,
      num_epochs = num_epochs,
      shuffle = shuffle,
      num_threads = 5)


def train_and_eval(model_dir, train_steps, train_file_name, test_file_name):
  """Train and evaluate the model."""
  m = tf.estimator.LinearClassifier(model_dir = model_dir, feature_columns = base_columns)

  # set num_epochs to None to get infinite stream of data.
  m.train(
      input_fn = input_fn(train_file_name, num_epochs = None, shuffle = True),
      steps = train_steps)
  # set steps to None to run evaluation until all data consumed.
  results = m.evaluate(
      input_fn = input_fn(test_file_name, num_epochs = 1, shuffle = False),
      steps = None)
  print("model directory = %s" % model_dir)
  for key in sorted(results):
    print("%s: %s" % (key, results[key]))


FLAGS = None


def main(_):
  train_and_eval(FLAGS.model_dir, FLAGS.train_steps,
                 FLAGS.train_data, FLAGS.test_data)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--model_dir",
      type=str,
      required=True,
      help="Base directory for output models."
  )
  parser.add_argument(
      "--train_steps",
      type=int,
      default=2000,
      help="Number of training steps."
  )
  parser.add_argument(
      "--train_data",
      type=str,
      required=True,
      help="Path to the training data."
  )
  parser.add_argument(
      "--test_data",
      type=str,
      required=True,
      help="Path to the test data."
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

