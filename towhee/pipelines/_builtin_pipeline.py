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

from towhee.dc2 import pipe
from towhee.runtime import AutoPipes, AutoConfig


@AutoConfig.register
class MyConfig:
    """
    For UT
    """
    def __init__(self):
        self.param = 1


@AutoPipes.register
def pipeline(config):
    """
    For UT
    """
    return (
        pipe.input('num')
        .map('num', 'ret', lambda x: x + config.param)
        .output('ret')
    )
