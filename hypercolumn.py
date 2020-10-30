from tensorflow.keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications.vgg19 import VGG19
from keras.preprocessing import image
import keras
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")
import scipy.misc
#-of-an-intermediate-layer
from keras import backend as K
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import cv2
from numpy.random import random as rnd
from skimage.transform import resize
import geometry_utils
import poisson_matting
import settings
import sys
import os
import geometry_utils
from PIL import Image
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np
from datetime import datetime as dt
import tensorflow as tf
import sklearn.cluster as cluster
from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.patches as patches
class hypercolumn():
    
    def __init__(self):
        #設定模型s
        self.feature_size=50
        #self.model = keras.applications.InceptionResNetV2(include_top=False,weights='imagenet',input_shape=(448, 448, 3)) 
        self.model = VGG19(include_top=False, weights='imagenet',input_shape=(448, 448, 3)) 
        #for layer in layers:
            #print(layer.name)
        #self.model.summary()
        
        #選擇提取特徵的層
        layers_extract = [1,2,4,5,7,8,9,10,12,13,14,15,17,18,19,20] #19 Full
        #layers_extract = [1,2,4,5,7,8,9,11,12,13,15,16,17]#16
        #layers_extract = [3,11,19] #19 
        #layers_extract2 = [22]
        #layers_extract = [19,20]
        #layers_extract = [6,41,87,155,189] #ResNet50V2 -2
        #layers_extract = [6,36,82,150,185] #ResNet50V2 -3
        #layers_extract = [189] #ResNet50V2 -4
        
        #layers_extract = [3,6,14,24,34,44,64,84,104,131] #Xception -2
        #layers_extract = [24,44,64,84,104,131] #Xception -3
        
        #layers_extract = [40,258,592,777] #InceptionResNetV2-1
        #layers_extract = [777] #InceptionResNetV2-2
        
        layers = [self.model.layers[li].output for li in layers_extract]
        
        self.get_feature = K.function([self.model.layers[0].input],layers)
       
        self.Matting = poisson_matting.Poisson_Matting()
    
    def extract_hypercolumn(self, instance):
        
        #耗時0.12s
        start = dt.now()
        feature_maps = self.get_feature(instance)  
        
        print("卷積層計算耗時:", (dt.now() - start))
        hypercolumns = []
        
        #耗時0.06s
        
        for convmap in feature_maps:
            conv_out = convmap[0, :, :, :]
            feat_map = conv_out.transpose((2,0,1))
            
            mean = np.mean(feat_map)
            average = np.average(feat_map, axis=0)
            A = (average>mean)
            plt.imshow(A)
            plt.show()
            A = average * A
            upscaled =resize(A, (self.feature_size, self.feature_size), mode='constant', preserve_range=True)
            
            hypercolumns.append(upscaled)
            '''
            i=0
            for fmap in feat_map: 
                i =i+1
                if (i%10==0):
                    upscaled =resize(fmap, (self.feature_size, self.feature_size), mode='constant', preserve_range=True)
                    hypercolumns.append(upscaled)
            '''
        print('疊加特徵圖耗時:', (dt.now() - start))
        
        return np.asarray(hypercolumns)
    
    def do_hypercolumn(self,img):
        
        self.height, self.width, domain = img.shape
        
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        self.hc = self.extract_hypercolumn( [x] )
        
        self.hcf = self.hc.transpose(1,2,0)
        print(self.hcf.shape)
        avg_hc=np.average(self.hc, axis=0)
        

    def features(self, points, layers=None):
        """
        Extracts the features for a particular set of points.
        Call this function only after you have called `init_with_image`
        """
        
        n_points = points.shape[0]
        n_features = self.hcf.shape[2]
        features = np.zeros((n_points, n_features), dtype=np.float32)
        
        
        for i, point in enumerate(points):
            x=int((point[0]/self.width)*self.feature_size)
            y=int((point[1]/self.height)*self.feature_size)
            features[i, :] = self.hcf[y,x , :]
            
        return features


    def image_point_features(self, img_origin,img_gray, part_box, part_name):
        """
        Extracts a set of positive and negative features from points generated from the image for a particular part.
        This function calls `init_with_image` so no need to call that yourself.
        """
        img = self.do_hypercolumn(img_origin)
        
        box = geometry_utils.Box.box_from_img(img_origin)

        self.height, self.width, domain = img_origin.shape
        
        positive_points = part_box.generate_points_inside(param=settings.POISSON_PART_RADIUS[part_name], img=img_origin)
        
        negative_points = box.generate_points_inside(param=settings.POISSON_NEGATIVE_RADIUS, img=img_origin)
        
        negative_points = geometry_utils.Box.filter_points(negative_points, part_box,img_gray)
        
        start = dt.now()
        #mat_img = self.Matting.Matting(img_origin,img_gray)
        #print('Possion Matting耗時:', (dt.now() - start))
        
        negative_points, positive_points = geometry_utils.Box.filter_points_negtive(negative_points, positive_points, img_gray)
        
        '''
        plt.scatter(*zip(*negative_points), color='r', alpha=0.6, lw=0)
        plt.scatter(*zip(*positive_points), color='b', alpha=0.6, lw=0)
        plt.savefig('point.png')#儲存圖片
        plt.show()
        '''
        return self.features(positive_points), self.features(negative_points)
