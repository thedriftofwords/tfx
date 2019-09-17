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
"""Definition of Airflow TFX runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from airflow import models

from typing import Any, Dict, Optional, Text

from tfx.orchestration import pipeline
from tfx.orchestration import tfx_runner
from tfx.orchestration.airflow import airflow_component
from tfx.orchestration.config import config_utils
from tfx.orchestration.config import pipeline_config


class AirflowPipelineConfig(pipeline_config.PipelineConfig):
  """Pipeline config for AirflowDagRunner."""

  def __init__(self, airflow_dag_config: Dict[Text, Any] = None, **kwargs):
    """Creates an instance of AirflowPipelineConfig.

    Args:
      airflow_dag_config: Configs of Airflow DAG model. See
        https://airflow.apache.org/_api/airflow/models/dag/index.html#airflow.models.dag.DAG
          for the full spec.
      **kwargs: keyword args for PipelineConfig.
    """

    super(AirflowPipelineConfig, self).__init__(kwargs)
    self.airflow_dag_config = airflow_dag_config or {}


class AirflowDagRunner(tfx_runner.TfxRunner):
  """Tfx runner on Airflow."""

  def __init__(self, config: Optional[AirflowPipelineConfig] = None):
    """Creates an instance of AirflowDagRunner.

    Args:
      config: Optional Airflow pipeline config for customizing the launching of
        each component.
    """
    if config and not isinstance(config, AirflowPipelineConfig):
      raise TypeError('config must be type of AirflowPipelineConfig.')
    super(AirflowDagRunner, self).__init__(config)

  def run(self, tfx_pipeline: pipeline.Pipeline):
    """Deploys given logical pipeline on Airflow.

    Args:
      tfx_pipeline: Logical pipeline containing pipeline args and components.

    Returns:
      An Airflow DAG.
    """

    # Merge airflow-specific configs with pipeline args
    airflow_dag = models.DAG(
        dag_id=tfx_pipeline.pipeline_info.pipeline_name,
        **self._config.airflow_dag_config)
    if 'tmp_dir' not in tfx_pipeline.additional_pipeline_args:
      tmp_dir = os.path.join(tfx_pipeline.pipeline_info.pipeline_root, '.temp',
                             '')
      tfx_pipeline.additional_pipeline_args['tmp_dir'] = tmp_dir

    component_impl_map = {}
    for tfx_component in tfx_pipeline.components:

      (component_launcher_class,
       component_config) = config_utils.find_component_launch_info(
           self._config, tfx_component)
      current_airflow_component = airflow_component.AirflowComponent(
          airflow_dag,
          component=tfx_component,
          component_launcher_class=component_launcher_class,
          pipeline_info=tfx_pipeline.pipeline_info,
          enable_cache=tfx_pipeline.enable_cache,
          metadata_connection_config=tfx_pipeline.metadata_connection_config,
          beam_pipeline_args=tfx_pipeline.beam_pipeline_args,
          additional_pipeline_args=tfx_pipeline.additional_pipeline_args,
          component_config=component_config)
      component_impl_map[tfx_component] = current_airflow_component
      for upstream_node in tfx_component.upstream_nodes:
        assert upstream_node in component_impl_map, ('Components is not in '
                                                     'topological order')
        current_airflow_component.set_upstream(
            component_impl_map[upstream_node])

    return airflow_dag
