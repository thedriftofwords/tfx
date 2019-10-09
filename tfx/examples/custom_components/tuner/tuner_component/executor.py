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
"""Generic TFX tuner executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import tensorflow as tf
from typing import Any, Dict, List, Text
from tensorflow_metadata.proto.v0 import schema_pb2
from tfx import types
from tfx.components.base import base_executor
from tfx.types import artifact_utils
from tfx.utils import import_utils
from tfx.utils import io_utils

# Default file name for generated best hyperparameters file.
_DEFAULT_FILE_NAME = 'best_hparams.txt'


class Executor(base_executor.BaseExecutor):
  """TFX Tuner component executor."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    train_path = artifact_utils.get_split_uri(input_dict['examples'], 'train')
    eval_path = artifact_utils.get_split_uri(input_dict['examples'], 'eval')
    schema_file = io_utils.get_only_uri_in_dir(
        artifact_utils.get_single_uri(input_dict['schema']))
    schema = io_utils.parse_pbtxt_file(schema_file, schema_pb2.Schema())

    tuning_state_path = artifact_utils.get_single_uri(
        output_dict['tuning_state'])
    tf.logging.info('Tuning State will be written to %s.' % tuning_state_path)
    tuner_fn = import_utils.import_func_from_source(
        exec_properties['module_file'], 'tuner_fn')
    tuner_spec = tuner_fn(tuning_state_path, train_path, eval_path, schema)
    tuner = tuner_spec['tuner']

    tuner.search_space_summary()
    # TODO(jyzhao): make epochs configurable.
    tuner.search(
        tuner_spec['train_dataset'],
        epochs=5,
        validation_data=tuner_spec['eval_dataset'])
    tuner.results_summary()

    best_hparams = tuner.get_state()['best_trial']['hyperparameters']
    best_hparams_path = os.path.join(
        artifact_utils.get_single_uri(output_dict['best_hparams']),
        _DEFAULT_FILE_NAME)
    io_utils.write_string_file(best_hparams_path, json.dumps(best_hparams))
    tf.logging.info('Best HParams is written to %s.' % best_hparams_path)

    # TODO(jyzhao): export best tuning model.
