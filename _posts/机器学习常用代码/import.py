import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import catboost as cb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, ShuffleSplit,StratifiedKFold,TimeSeriesSplit,KFold,GroupKFold,train_test_split,GroupShuffleSplit,StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score,mean_squared_error,mean_absolute_error,log_loss,confusion_matrix,accuracy_score
import sqlite3
import xgboost as xgb
import datetime
from sklearn.linear_model import LogisticRegression
from scipy.stats import pearsonr
import gc
from sklearn.model_selection import TimeSeriesSplit
#from bayes_opt import BayesianOptimization
import re
from string import punctuation
from scipy.spatial import Voronoi
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from tqdm.notebook import tqdm
#from numba import jit
from collections import Counter
import json
import joblib
import multiprocessing
import time
import keras
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K
from keras import models
from keras import layers
from keras.layers import *
from keras.models import *
from keras.callbacks import *
import math 
import logging
from scipy.sparse import csr_matrix,hstack
import scipy
# pd.set_option('display.max_rows', 100)
# pd.set_option('display.max_columns', 25)

# pd.set_option('max_columns', None)
# pd.set_option('max_rows', 300)
# pd.set_option('max_colwidth', 200)