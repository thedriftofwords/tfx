# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM ubuntu:16.04
LABEL maintainer="tensorflow-extended-dev@googlegroups.com"

# Need additional ppa since python 3.6 and protobuf 3 
# are not part of Ubuntu 16.04 by default.
RUN apt-get update -y && \
  apt-get install --no-install-recommends -y -q software-properties-common && \
  add-apt-repository ppa:deadsnakes/ppa && \
  add-apt-repository ppa:maarten-fonville/protobuf && \
  apt-get update -y && \
  apt-get install --no-install-recommends -y -q \
  build-essential \
  ca-certificates \
  libsnappy-dev \
  protobuf-compiler \
  libprotobuf-dev \
  python3.6-dev \
  python3-pip \
  python3-setuptools \
  python3-virtualenv \
  python3-wheel \
  wget \
  unzip \
  git && \
  add-apt-repository -r ppa:deadsnakes/ppa && \
  add-apt-repository -r ppa:maarten-fonville/protobuf && \
  apt-get autoclean && \
  apt-get autoremove --purge && \
  update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1

ENV VIRTUAL_ENV=/opt/venv
RUN python3.6 -m virtualenv --python=/usr/bin/python3.6 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# docker build command should be run under root directory of github checkout.
ENV TFX_SRC_DIR=/tfx-src
ADD . ${TFX_SRC_DIR}
WORKDIR ${TFX_SRC_DIR}
RUN python setup.py bdist_wheel
RUN CFLAGS=$(/usr/bin/python3.6-config --cflags) pip install $(find dist -name "tfx-*.whl")[docker_image]

# Patch http.py in googleapiclient and base_api.py in apitools
# to use our own UserAgent.
RUN patch /opt/venv/lib/python3.6/site-packages/googleapiclient/http.py \
  /tfx-src/tfx/tools/docker/patches/http.patch && \
  patch /opt/venv/lib/python3.6/site-packages/apitools/base/py/base_api.py \
  /tfx-src/tfx/tools/docker/patches/base_api.patch

ARG gather_third_party_licenses=true
RUN if [ "${gather_third_party_licenses}" = "true" ]; then \
  /tfx-src/tfx/tools/docker/license.sh \
  /tfx-src/tfx/tools/docker/third_party_licenses.csv \
  /third_party/licenses; \
  fi

# TODO(b/139532550): Consolidate container entrypoint with Kubeflow runner.
ENTRYPOINT ["python3.6", "/tfx-src/tfx/scripts/run_executor.py"]
