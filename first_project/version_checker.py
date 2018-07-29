#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @FileName: version_checker.py
 @Desc:
 @Author: yuzhongchun
 @Date: 2018-07-01 12:01:52
 @Last Modified by: yuzhongchun
 @Last Modified time: 2018-07-01 12:01:52
"""

import sys
import scipy
import numpy
import pandas
import matplotlib
import seaborn
import sklearn

print('Python: {}'.format(sys.version))
print('scipy: {}'.format(scipy.__version__))
print('numpy: {}'.format(numpy.__version__))
print('pandas: {}'.format(pandas.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('seaborn: {}'.format(seaborn.__version__))
print('sklearn: {}'.format(sklearn.__version__))
