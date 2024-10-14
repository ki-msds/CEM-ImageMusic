def runmodel_a(modelname, images, temp_path, main_path):
  import os
  import numpy as np
  import pandas as pd
  from matplotlib import pyplot as plt
  from sklearn.preprocessing import MinMaxScaler
  import seaborn as sns

  from tensorflow.keras.utils import to_categorical
  from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
  from keras.preprocessing.image import array_to_img, img_to_array
  from tensorflow.keras.preprocessing.image import ImageDataGenerator

  import keras.models
  from keras.layers import Input, Flatten, Dense, Conv2D, BatchNormalization, Dropout, LeakyReLU, Softmax, MaxPooling2D
  from keras.models import Model

  from PIL import Image
  from keras.preprocessing.image import load_img
  from keras.preprocessing.image import img_to_array
  from keras.preprocessing.image import array_to_img

  model = keras.models.load_model(modelname)

  os.chdir(temp_path)

  filenames = np.array(images)

  #convert images to arrays
  img_arrays_full = np.zeros((0,100,125,3))
  image_dataset_full = []
  for filename in filenames:
    #we rescale the images to 100x125 to lessen load
    image = load_img(filename, target_size=(100,125))
    image_array = img_to_array(image)  
    img_arrays_full = np.append(img_arrays_full, np.array([image_array]),
                                axis = 0)
  
  CLASSES = np.array(['Happy',
                    'Angry',
                    'Sad',
                    'Relaxed'])
  
  #run model and predict
  predictions = model.predict(img_arrays_full)

  #argmax predictions
  #predictions = CLASSES[np.argmax(predictions, axis = -1)]
  predictions = np.argmax(predictions, axis = -1)
  
  os.chdir(main_path)

  return predictions
