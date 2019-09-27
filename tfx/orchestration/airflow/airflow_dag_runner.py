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

from typing import Any, Dict, List, Text

from tfx.orchestration import pipeline
from tfx.orchestration import tfx_runner
from tfx.orchestration.airflow import airflow_component
from tfx.orchestration.config import base_platform_config
from tfx.orchestration.launcher import in_process_component_launcher


class AirflowDagRunner(tfx_runner.TfxRunner):
  """Tfx runner on Airflow.

  The supported launcher classes are (in the order of preference):
  `in_process_component_launcher.InProcessComponentLauncher`.
  """

  SUPPORTED_LAUNCHER_CLASSES = [
      in_process_component_launcher.InProcessComponentLauncher
  ]

  def __init__(self,
               airflow_dag_config: Dict[Text, Any] = None,
               platform_configs: Dict[
                   Text, List[base_platform_config.BasePlatformConfig]] = None):
    """Creates an instance of AirflowDagRunner.

    Args:
      airflow_dag_config: Configs of Airflow DAG model. See
        https://airflow.apache.org/_api/airflow/models/dag/index.html#airflow.models.dag.DAG
          for the full spec.
      platform_configs: Optional platform configs for customizing the launching
        of each component. The key is the component ID and the value is a list
        platform configs.
    """
    super(AirflowDagRunner, self).__init__(platform_configs)
    self._airflow_dag_config = airflow_dag_config or {}

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
        **self._airflow_dag_config)
    if 'tmp_dir' not in tfx_pipeline.additional_pipeline_args:
      tmp_dir = os.path.join(tfx_pipeline.pipeline_info.pipeline_root, '.temp',
                             '')
      tfx_pipeline.additional_pipeline_args['tmp_dir'] = tmp_dir

    component_impl_map = {}
    for tfx_component in tfx_pipeline.components:
      (component_launcher_class,
       platform_config) = self.find_component_launch_info(tfx_component)
      current_airflow_component = airflow_component.AirflowComponent(
          airflow_dag,
          component=tfx_component,
          component_launcher_class=component_launcher_class,
          pipeline_info=tfx_pipeline.pipeline_info,
          enable_cache=tfx_pipeline.enable_cache,
          metadata_connection_config=tfx_pipeline.metadata_connection_config,
          beam_pipeline_args=tfx_pipeline.beam_pipeline_args,
          additional_pipeline_args=tfx_pipeline.additional_pipeline_args,
          platform_config=platform_config)
      component_impl_map[tfx_component] = current_airflow_component
      for upstream_node in tfx_component.upstream_nodes:
        assert upstream_node in component_impl_map, ('Components is not in '
                                                     'topological order')
        current_airflow_component.set_upstream(
            component_impl_map[upstream_node])

    return airflow_dag
