# Copyright 2021 Zilliz. All rights reserved.
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
import numpy as np


class NumpyFormat:
    """
    NumpyFormat:
    """

    def __init__(self, data: 'ndarray'):
        self._data = data

    def to_dict(self):
        return {
            '_NP': True,
            'dtype': NumpyFormat.from_numpyt_type(self._data.dtype),
            'data': self._data.tolist()
        }

    @staticmethod
    def from_dict(dct):
        return np.array(dct['data'], NumpyFormat.to_numpy_type(dct['dtype']))

    @staticmethod
    def from_numpyt_type(dtype):
        if dtype == np.uint8:
            return 1
        if dtype == np.uint16:
            return 2
        if dtype == np.uint32:
            return 3
        if dtype == np.uint64:
            return 4
        if dtype == np.int8:
            return 5
        if dtype == np.int16:
            return 6
        if dtype == np.int32:
            return 7
        if dtype == np.int64:
            return 8
        if dtype == np.float16:
            return 9
        if dtype == np.float32:
            return 10
        if dtype == np.float64:
            return 11
        raise ValueError('Unsupport numpy type: %s ' % str(dtype))

    @staticmethod
    def to_numpy_type(dtype):
        if dtype == 1:
            return np.uint8
        if dtype == 2:
            return np.uint16
        if dtype == 3:
            return np.uint32
        if dtype == 4:
            return np.uint64
        if dtype == 5:
            return np.int8
        if dtype == 6:
            return np.int16
        if dtype == 7:
            return np.int32
        if dtype == 8:
            return np.int64
        if dtype == 9:
            return np.float16
        if dtype == 10:
            return np.float32
        if dtype == 11:
            return np.float64
        raise ValueError('Unsupport numpy type code %s' % dtype)
