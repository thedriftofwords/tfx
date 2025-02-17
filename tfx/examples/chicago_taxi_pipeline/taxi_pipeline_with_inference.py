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
"""Chicago taxi example pipeline for training and offline inference."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os

import absl
from absl import logging
from typing import Text

from tfx.components.bulk_inferrer.component import BulkInferrer
from tfx.components.evaluator.component import Evaluator
from tfx.components.example_gen.csv_example_gen.component import CsvExampleGen
from tfx.components.example_validator.component import ExampleValidator
from tfx.components.model_validator.component import ModelValidator
from tfx.components.schema_gen.component import SchemaGen
from tfx.components.statistics_gen.component import StatisticsGen
from tfx.components.trainer.component import Trainer
from tfx.components.transform.component import Transform
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.proto import bulk_inferrer_pb2
from tfx.proto import evaluator_pb2
from tfx.proto import example_gen_pb2
from tfx.proto import trainer_pb2
from tfx.utils.dsl_utils import external_input

_pipeline_name = 'chicago_taxi_with_inference'

# This example assumes that the taxi data is stored in ~/taxi/data and the
# taxi utility function is in ~/taxi.  Feel free to customize this as needed.
_taxi_root = os.path.join(os.environ['HOME'], 'taxi')
_training_data_root = os.path.join(_taxi_root, 'data', 'simple')
_inference_data_root = os.path.join(_taxi_root, 'data', 'unlabelled')
# Python module file to inject customized logic into the TFX components. The
# Transform and Trainer both require user-defined functions to run successfully.
_module_file = os.path.join(_taxi_root, 'taxi_utils.py')

# Directory and data locations.  This example assumes all of the chicago taxi
# example code and metadata library is relative to $HOME, but you can store
# these files anywhere on your local filesystem.
_tfx_root = os.path.join(os.environ['HOME'], 'tfx')
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)
# Sqlite ML-metadata db path.
_metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name,
                              'metadata.db')


def _create_pipeline(pipeline_name: Text, pipeline_root: Text,
                     training_data_root: Text, inference_data_root: Text,
                     module_file: Text,
                     metadata_path: Text,
                     direct_num_workers: int) -> pipeline.Pipeline:
  """Implements the chicago taxi pipeline with TFX."""
  training_examples = external_input(training_data_root)

  # Brings training data into the pipeline or otherwise joins/converts
  # training data.
  training_example_gen = CsvExampleGen(
      input_base=training_examples, instance_name='training_example_gen')

  # Computes statistics over data for visualization and example validation.
  statistics_gen = StatisticsGen(
      input_data=training_example_gen.outputs['examples'])

  # Generates schema based on statistics files.
  infer_schema = SchemaGen(
      stats=statistics_gen.outputs['output'], infer_feature_shape=False)

  # Performs anomaly detection based on statistics and data schema.
  validate_stats = ExampleValidator(
      stats=statistics_gen.outputs['output'],
      schema=infer_schema.outputs['output'])

  # Performs transformations and feature engineering in training and serving.
  transform = Transform(
      input_data=training_example_gen.outputs['examples'],
      schema=infer_schema.outputs['output'],
      module_file=module_file)

  # Uses user-provided Python function that implements a model using TF-Learn.
  trainer = Trainer(
      module_file=module_file,
      transformed_examples=transform.outputs['transformed_examples'],
      schema=infer_schema.outputs['output'],
      transform_output=transform.outputs['transform_output'],
      train_args=trainer_pb2.TrainArgs(num_steps=10000),
      eval_args=trainer_pb2.EvalArgs(num_steps=5000))

  # Uses TFMA to compute a evaluation statistics over features of a model.
  model_analyzer = Evaluator(
      examples=training_example_gen.outputs['examples'],
      model_exports=trainer.outputs['output'],
      feature_slicing_spec=evaluator_pb2.FeatureSlicingSpec(specs=[
          evaluator_pb2.SingleSlicingSpec(
              column_for_slicing=['trip_start_hour'])
      ]))

  # Performs quality validation of a candidate model (compared to a baseline).
  model_validator = ModelValidator(
      examples=training_example_gen.outputs['examples'],
      model=trainer.outputs['output'])

  inference_examples = external_input(inference_data_root)

  # Brings inference data into the pipeline.
  inference_example_gen = CsvExampleGen(
      input_base=inference_examples,
      output_config=example_gen_pb2.Output(
          split_config=example_gen_pb2.SplitConfig(
              splits=[example_gen_pb2.SplitConfig.Split(
                  name='unlabelled', hash_buckets=100)])),
      instance_name='inference_example_gen')

  # Performs offline batch inference over inference examples.
  bulk_inferrer = BulkInferrer(
      examples=inference_example_gen.outputs['examples'],
      model_export=trainer.outputs['output'],
      model_blessing=model_validator.outputs['blessing'],
      # Empty data_spec.example_splits will result in using all splits.
      data_spec=bulk_inferrer_pb2.DataSpec(),
      model_spec=bulk_inferrer_pb2.ModelSpec())

  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[
          training_example_gen, inference_example_gen, statistics_gen,
          infer_schema, validate_stats, transform, trainer, model_analyzer,
          model_validator, bulk_inferrer
      ],
      enable_cache=True,
      metadata_connection_config=metadata.sqlite_metadata_connection_config(
          metadata_path),
      # TODO(b/141578059): The multi-processing API might change.
      beam_pipeline_args=['--direct_num_workers=%d' % direct_num_workers])

try:
  parallelism = multiprocessing.cpu_count()
except NotImplementedError:
  parallelism = 1
absl.logging.info('Using %d process(es) for Beam pipeline execution.' %
                  parallelism)

# To run this pipeline from the python CLI:
#   $python taxi_pipeline_offline_inference.py
if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  BeamDagRunner().run(
      _create_pipeline(
          pipeline_name=_pipeline_name,
          pipeline_root=_pipeline_root,
          training_data_root=_training_data_root,
          inference_data_root=_inference_data_root,
          module_file=_module_file,
          metadata_path=_metadata_path,
          direct_num_workers=parallelism))
