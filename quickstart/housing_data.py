from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves.urllib.request import urlopen
import itertools
import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

#first download data set if not exist locally
BOSTON_TRAINING = "boston_train.csv"
BOSTON_TRAINING_URL = "http://download.tensorflow.org/data/boston_train.csv"

BOSTON_TEST = "boston_test.csv"
BOSTON_TEST_URL = "http://download.tensorflow.org/data/boston_test.csv"

BOSTON_PREDICT = "boston_predict.csv"
BOSTON_PREDICT_URL = "http://download.tensorflow.org/data/boston_predict.csv"

if not os.path.exists(BOSTON_TRAINING):
  raw = urlopen(BOSTON_TRAINING_URL).read()
  with open(BOSTON_TRAINING,'wb') as f:
    f.write(raw)

if not os.path.exists(BOSTON_TEST):
  raw = urlopen(BOSTON_TEST_URL).read()
  with open(BOSTON_TEST,'wb') as f:
    f.write(raw)
if not os.path.exists(BOSTON_PREDICT):
  raw = urlopen(BOSTON_PREDICT_URL).read()
  with open(BOSTON_PREDICT,'wb') as f:
    f.write(raw)

# setter opp that sweet sweet pandas datasets
COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"]
FEATURES = ["crim", "zn", "indus", "nox", "rm",
            "age", "dis", "tax", "ptratio"]
LABEL = "medv"

training_set = pd.read_csv("boston_train.csv", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)
test_set = pd.read_csv("boston_test.csv", skipinitialspace=True,
                       skiprows=1, names=COLUMNS)
prediction_set = pd.read_csv("boston_predict.csv", skipinitialspace=True,
                             skiprows=1, names=COLUMNS)
#fordi alle kollonner har ekte tall i seg, fortell tensorflow dette og navnet på features
feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

#lagrinsplass
tempLagringsPlass=os.path.join(os.getenv('TEST_TMPDIR', '/AppData/Local/Temp/'),
                           'tensorflow/boston')
if tf.gfile.Exists(tempLagringsPlass):
    tf.gfile.DeleteRecursively(tempLagringsPlass)
tf.gfile.MakeDirs(tempLagringsPlass)

#ANN modellen
regressor = tf.estimator.DNNRegressor(hidden_units=[10,10],
                                      feature_columns=feature_cols,
                                      model_dir=tempLagringsPlass)

def get_input_fn(data_set, num_epochs=None, shuffle=False):
    return tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k:data_set[k].values for k in FEATURES}),
        y=pd.Series(data_set[LABEL].values),
        num_epochs=num_epochs,
        shuffle=shuffle
    )