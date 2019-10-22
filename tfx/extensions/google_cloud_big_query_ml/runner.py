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
"""Helper class to start TFX training jobs on BQML."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import absl
from googleapiclient import errors
from typing import Any, Dict, Text
from google.cloud import bigquery


def deploy_model_for_bigquery_ml_serving(serving_path: Text,
                                         model_version: Text,
                                         bqml_serving_args: Dict[Text, Any]):
  """Deploys a model for serving with big query ML.

  Args:
    serving_path: The path to the model. Must be a GCS URI.
    model_version: Version of the model being deployed. Can be same as what is
      currently being served if deployment mode configuration is set to
      "CREATE_OR_REPLACE".
    bqml_serving_args: Dictionary containing arguments for pushing to BQML. For
      the full set of parameters supported, refer to
      https://cloud.google.com/ml-engine/reference/rest/v1/projects.models.versions#Version

  Raises:
    RuntimeError: If an error is encountered when trying to push.
  """
  absl.logging.info(
      'Deploying to model with version {} to BigQuery ML for serving: {}'
      .format(model_version, bqml_serving_args))

  model_name = bqml_serving_args['model_name']
  project_id = bqml_serving_args['project_id']
  bq_dataset_id = bqml_serving_args['bq_dataset_id']

  # Setting'CREATE OR REPLACE MODEL' as default mode, the side effect is that a
  # model can be overwritten silently with no option to role back.
  bq_model_create_mode = bqml_serving_args.get('bq_model_create_mode',
                                               'CREATE OR REPLACE MODEL')
  client = bigquery.Client()

  # Check if dataset exists if not create it
  try:
    client.get_dataset(bq_dataset_id)
  except errors.HttpError as validate_ex:
    if 'Not found: Dataset' not in validate_ex.args[0]:
      raise RuntimeError(
          'BigQuery dataset validation failed: {}'.format(validate_ex))

    # create a new dataset, use US if location is not specificed
    dataset = bigquery.Dataset('{}.{}'.format(project_id, bq_dataset_id))
    dataset.location = bqml_serving_args.get('bq_dataset_location', 'US')

    try:
      dataset = client.create_dataset(dataset)
    except errors.HttpError as create_ex:
      raise RuntimeError(
          'BigQuery dataset creation failed: {}'.format(create_ex))

  # TODO(chavoshi) BigQuery ML file size should not exceed 250 mb, this check is
  # requirement should be validated in a dedicated infra validation step.
  query = ("""
  {} `{}.{}.{}`
  OPTIONS (model_type='tensorflow',
           model_path='{}')
    """.format(bq_model_create_mode, project_id, bq_dataset_id, model_name,
               serving_path))

  try:
    query_job = client.query(query)
    query_job.result()
  except errors.HttpError as ex:
    absl.logging.error(ex)
    raise RuntimeError('BQML Push failed: {}'.format(ex))

  absl.logging.info(
      'Successfully deployed model {} with version {}, serving from {}'.format(
          model_name, model_version, serving_path))
