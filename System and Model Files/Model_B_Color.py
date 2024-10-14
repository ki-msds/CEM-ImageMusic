#Receives a list of images
def colourpaletteextraction(images, temp_path, main_path):
  import os
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

  #get images
  os.chdir(temp_path)

  filenames = images
  img_arrays = np.zeros((0,100,125,3))
  image_dataset_full = []

  for filename in filenames:
    #we rescale the images to 100x125
    image = load_img(filename, target_size=(100,125))
    image_array = img_to_array(image)  
    img_arrays = np.append(img_arrays, np.array([image_array]),
                           axis = 0)
  
  colours = 10

  quadrantresults = []

  #Flattening
  for image in img_arrays:
    imagedf = pd.DataFrame()
    imagedf['r'] = pd.Series(image[:,:,0].flatten())
    imagedf['g'] = pd.Series(image[:,:,1].flatten())
    imagedf['b'] = pd.Series(image[:,:,2].flatten())
    imagedf['r_whiten'] = whiten(imagedf['r'])
    imagedf['g_whiten'] = whiten(imagedf['g'])
    imagedf['b_whiten'] = whiten(imagedf['b'])
    
    #K-Means
    cluster_centers, distortion = kmeans(imagedf[['r_whiten',
                                                  'g_whiten',
                                                  'b_whiten']],
                                         colours)
    
    r_std, g_std, b_std = imagedf[['r','g','b']].std() 
    palette = []
    for color in cluster_centers:
      sr, sg, sb = color
      palette.append([int(sr*r_std), int(sg*g_std), int(sb*b_std)])

    palette_ar = np.array(palette)

    testavg = np.mean([palette_ar], axis=(0,1)).round().astype(int)
    testavg = testavg.tolist()

    h = testavg[0]
    hyp = testavg[1]

    wheelplc = []
    if h > 90 and h <= 180:
      ha = 180 - h
      wheelplc.append(ha)
      wheelplc.append(1)
    elif h > 180 and h <= 270:
      ha = 270 - h
      wheelplc.append(ha)
      wheelplc.append(2)
    elif h > 270 and h <= 360:
      ha = 360 - h
      wheelplc.append(ha)
      wheelplc.append(3)
    else:
      wheelplc.append(h)
      wheelplc.append(0)

    wheel_x = hyp * math.cos(math.radians(wheelplc[0]))
    wheel_y = hyp * math.sin(math.radians(wheelplc[0]))

    #NOTE we are remapping according to affect interpretation (See RRL)

    if wheelplc[1] == 1:
      wheel_y = -wheel_y
    elif wheelplc[1] == 2:
      wheel_x = -wheel_x
      wheel_y = -wheel_y
    elif wheelplc[1] == 3:
      wheel_x = -wheel_x

    wheelplc.append(wheel_x)
    wheelplc.append(wheel_y)

    quadrantresults.append(wheelplc[1])

  quadrantresults = np.array(quadrantresults)

  os.chdir(main_path)
  
  return quadrantresults


