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
"""Utilities to dump and load Jsonable object to/from JSONs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import importlib
import inspect
import json

from six import with_metaclass
from typing import Any, Dict, List, Text, Type, Union

from google.protobuf import json_format
from google.protobuf import message

# This is the special key to indicate the serialized object type.
# Depending on which, the utility knows how to deserialize it back to its
# original type.
_TFX_OBJECT_TYPE_KEY = '__tfx_object_type__'
_MODULE_KEY = '__module__'
_CLASS_KEY = '__class__'
_PROTO_VALUE_KEY = '__proto_value__'


class _ObjectType(object):
  """Internal class to hold supported types."""
  # Indicates that the JSON dictionary is an instance of Jsonable type.
  # The dictionary has the states of the object and the object type info is
  # stored as __module__ and __class__ fields.
  JSONABLE = 'jsonable'
  # Indicates that the JSON dictionary is a python class.
  # The class info is stored as __module__ and __class__ fields in the
  # dictionary.
  CLASS = 'class'
  # Indicates that the JSON dictionary is an instance of a proto.Message
  # subclass. The class info of the proto python class is stored as __module__
  # and __class__ fields in the dictionary. The serialized value of the proto is
  # stored in the dictionary with key of _PROTO_VALUE_KEY.
  PROTO = 'proto'


class Jsonable(with_metaclass(abc.ABCMeta, object)):
  """Base class for serializing and deserializing objects to/from JSON.

  The default implementation assumes that the subclass can be restored by
  updating `self.__dict__` without invoking `self.__init__` function.. If the
  subclass cannot hold the assumption, it should
  override `to_json_dict` and `from_json_dict` to customize the implementation.
  """

  def to_json_dict(self) -> Dict[Text, Any]:
    """Convert from an object to a JSON serializable dictionary."""
    return self.__dict__

  @classmethod
  def from_json_dict(cls, dict_data: Dict[Text, Any]) -> Any:
    """Convert from dictionary data to an object."""
    instance = cls.__new__(cls)
    instance.__dict__ = dict_data
    return instance


JsonableValue = Union[bool, bytes, float, int, Jsonable, message.Message, Text,
                      Type]
JsonableList = List[JsonableValue]
JsonableDict = Dict[Union[bytes, Text], Union[JsonableValue, JsonableList]]
JsonableType = Union[JsonableValue, JsonableList, JsonableDict]


class _DefaultEncoder(json.JSONEncoder):
  """Default JSON Encoder which encodes Jsonable object to JSON."""

  def default(self, obj: Any) -> Any:
    if isinstance(obj, Jsonable):
      dict_data = {
          _TFX_OBJECT_TYPE_KEY: _ObjectType.JSONABLE,
          _MODULE_KEY: obj.__class__.__module__,
          _CLASS_KEY: obj.__class__.__name__,
      }
      dict_data.update(obj.to_json_dict())
      return dict_data

    if inspect.isclass(obj):
      return {
          _TFX_OBJECT_TYPE_KEY: _ObjectType.CLASS,
          _MODULE_KEY: obj.__module__,
          _CLASS_KEY: obj.__name__,
      }

    if isinstance(obj, message.Message):
      return {
          _TFX_OBJECT_TYPE_KEY: _ObjectType.PROTO,
          _MODULE_KEY: obj.__class__.__module__,
          _CLASS_KEY: obj.__class__.__name__,
          _PROTO_VALUE_KEY: json_format.MessageToJson(obj, sort_keys=True)
      }

    return super(_DefaultEncoder, self).default(obj)


class _DefaultDecoder(json.JSONDecoder):
  """Default JSON Decoder which decodes JSON to Jsonable object."""

  def __init__(self, *args, **kwargs):
    super(_DefaultDecoder, self).__init__(
        object_hook=self._dict_to_object, *args, **kwargs)

  def _dict_to_object(self, dict_data: Dict[Text, Any]) -> Any:
    """Converts a dictionary to an object."""
    if _TFX_OBJECT_TYPE_KEY not in dict_data:
      return dict_data

    object_type = dict_data.pop(_TFX_OBJECT_TYPE_KEY)

    def _extract_class(d):
      module_name = d.pop(_MODULE_KEY)
      class_name = d.pop(_CLASS_KEY)
      return getattr(importlib.import_module(module_name), class_name)

    if object_type == _ObjectType.JSONABLE:
      jsonable_class_type = _extract_class(dict_data)
      if not issubclass(jsonable_class_type, Jsonable):
        raise ValueError('Class %s must be a subclass of Jsonable' %
                         jsonable_class_type)
      return jsonable_class_type.from_json_dict(dict_data)

    if object_type == _ObjectType.CLASS:
      return _extract_class(dict_data)

    if object_type == _ObjectType.PROTO:
      proto_class_type = _extract_class(dict_data)
      if not issubclass(proto_class_type, message.Message):
        raise ValueError('Class %s must be a subclass of proto.Message' %
                         proto_class_type)
      if _PROTO_VALUE_KEY not in dict_data:
        raise ValueError('Missing proto value in json dict')
      return json_format.Parse(dict_data[_PROTO_VALUE_KEY], proto_class_type())


def dumps(obj: Any) -> Text:
  """Dumps an object to JSON with Jsonable encoding."""
  return json.dumps(obj, cls=_DefaultEncoder, sort_keys=True)


def loads(s: Text) -> Any:
  """Loads a JSON into an object with Jsonable decoding."""
  return json.loads(s, cls=_DefaultDecoder)
