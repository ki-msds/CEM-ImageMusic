import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import glob
from music21 import converter, instrument, note, chord, stream
from music21 import *
import tensorflow as tf
import keras
from keras import *
from keras import backend as K

from tensorflow.keras import (
    layers,
    models,
    callbacks,
    utils,
    metrics,
    optimizers,
)

from tensorflow.keras.layers import Input, Dense, Reshape, Dropout,Conv2D, Conv2DTranspose, LSTM, Bidirectional, Embedding, Concatenate, Flatten
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from tqdm.notebook import tqdm

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import image as img

from PIL import Image

from scipy.cluster.vq import whiten
from scipy.cluster.vq import kmeans
import cv2

import colorsys

import math

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
