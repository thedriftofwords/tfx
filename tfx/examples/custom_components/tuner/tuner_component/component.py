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
"""TFX Tuner component definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Text

from tfx import types
from tfx.components.base import base_component
from tfx.components.base import executor_spec
from tfx.examples.custom_components.tuner.tuner_component import executor
from tfx.types import standard_artifacts
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ComponentSpec
from tfx.types.component_spec import ExecutionParameter


# TODO(jyzhao): move to tfx/types/standard_component_specs.py.
class TunerSpec(ComponentSpec):
  """ComponentSpec for TFX Tuner Component."""

  PARAMETERS = {
      'module_file': ExecutionParameter(type=(str, Text)),
  }
  INPUTS = {
      'examples': ChannelParameter(type=standard_artifacts.Examples),
      'schema': ChannelParameter(type=standard_artifacts.Schema),
  }
  OUTPUTS = {
      'tuning_state':
          ChannelParameter(type=standard_artifacts.TuningState),
      'model_export_path':
          ChannelParameter(type=standard_artifacts.Model),
      'study_best_hparams_path':
          ChannelParameter(type=standard_artifacts.BestHParams),
  }
  # TODO(b/139281215): these input / output names will be renamed in the future.
  # These compatibility aliases are provided for forwards compatibility.
  _OUTPUT_COMPATIBILITY_ALIASES = {
      'model': 'model_export_path',
      'best_hparams': 'study_best_hparams_path',
  }


class Tuner(base_component.BaseComponent):
  """A TFX component for model hyperparameter tuning."""

  SPEC_CLASS = TunerSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(
      self,
      examples: types.Channel = None,
      schema: types.Channel = None,
      # TODO(jyzhao): support optional tuner_fn param.
      module_file: Text = None,
      tuning_state: Optional[types.Channel] = None,
      model: Optional[types.Channel] = None,
      best_hparams: Optional[types.Channel] = None,
      instance_name: Optional[Text] = None):
    """Construct a Tuner component.

    Args:
      examples: A Channel of 'ExamplesPath' type, serving as the source of
        examples that are used in tuning (required). Transformed examples are
        not supported yet.
      schema:  A Channel of 'SchemaPath' type, serving as the schema of training
        and eval data.
      module_file: A path to python module file containing UDF KerasTuner
        definition (required).
      tuning_state: Optional 'TuningStatePath' channel for tuning state.
      model: Optional 'ModelExportPath' channel for result of best model.
      best_hparams: Optional 'StudyBestHParamsPath' channel for result of the
        best hparams.
      instance_name: Optional unique instance name. Necessary if multiple Tuner
        components are declared in the same pipeline.
    """
    tuning_state = tuning_state or types.Channel(
        type=standard_artifacts.TuningState,
        artifacts=[standard_artifacts.TuningState()])
    model = model or types.Channel(
        type=standard_artifacts.Model, artifacts=[standard_artifacts.Model()])
    best_hparams = best_hparams or types.Channel(
        type=standard_artifacts.BestHParams,
        artifacts=[standard_artifacts.BestHParams()])
    spec = TunerSpec(
        examples=examples,
        schema=schema,
        module_file=module_file,
        tuning_state=tuning_state,
        model_export_path=model,
        study_best_hparams_path=best_hparams)
    super(Tuner, self).__init__(spec=spec, instance_name=instance_name)
