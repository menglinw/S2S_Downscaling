import numpy as np
import sys
import os
import tensorflow as tf
import time
if '..' not in sys.path:
    sys.path.append('..')
from util_tools.data_loader import data_processer
import pandas as pd
from util_tools import downscale
import pandas as pd
from scipy import stats

from skgstat import Variogram, OrdinaryKriging
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


print('Successfully loaded packages!')