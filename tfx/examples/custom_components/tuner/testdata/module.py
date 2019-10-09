# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test module file for tuner's executor_test.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kerastuner import HyperParameters
from kerastuner import RandomSearch
import numpy as np
import tensorflow as tf
from tensorflow import keras

# TODO(jyzhao): unify with trainer's module file.


def _input_fn(file_path, schema, batch_size=200):  # pylint: disable=unused-argument
  """Generates features and labels for tuning/training.

  Args:
    file_path: input tfrecord data path.
    schema: Schema of the input data.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A tf.data.Dataset that contains features-labels tuples.
  """
  # TODO(jyzhao): make it real, converted from tfrecord.
  if 'train' in file_path:
    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))
  else:
    data = np.random.random((100, 32))
    labels = np.random.random((100, 10))

  dataset = tf.data.Dataset.from_tensor_slices((data, labels))
  dataset = dataset.batch(batch_size)
  return dataset


def _build_keras_model(hparams):
  """Creates Keras model for testing.

  Args:
    hparams: Holds HyperParameters for tuning.

  Returns:
    A Keras Model.
  """
  model = keras.Sequential()
  model.add(keras.layers.Dense(64, activation='relu', input_shape=(32,)))
  for _ in range(hparams.get('num_layers')):
    model.add(keras.layers.Dense(64, activation='relu'))
  model.add(keras.layers.Dense(10, activation='softmax'))
  model.compile(
      optimizer=keras.optimizers.Adam(hparams.get('learning_rate')),
      loss='categorical_crossentropy',
      metrics=['accuracy'])
  return model


def tuner_fn(working_dir, train_data_path, eval_data_path, schema):
  """Build the tuner using the Keras Tuner API.

  Args:
    working_dir: working dir for KerasTuner.
    train_data_path: path of training tfrecord data.
    eval_data_path: path of eval tfrecord data.
    schema: Schema of the input data.

  Returns:
    A dict of the following:
      - tuner: A KerasTuner that will be used for tuning.
      - train_dataset: A tf.data.Dataset of training data.
      - eval_dataset: A tf.data.Dataset of eval data.
  """
  hparams = HyperParameters()
  hparams.Choice('learning_rate', [1e-1, 1e-3])
  hparams.Int('num_layers', 2, 10)

  tuner = RandomSearch(
      _build_keras_model,
      max_trials=3,
      hyperparameters=hparams,
      allow_new_entries=False,
      objective='val_accuracy',
      directory=working_dir,
      project_name='test_project')

  return {
      'tuner': tuner,
      'train_dataset': _input_fn(train_data_path, schema, 32),
      'eval_dataset': _input_fn(eval_data_path, schema, 32),
  }
