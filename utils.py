# RCP209 Project - Implementation of Image Style Transfer
# Gilles Augustins - 22/02/2019

import numpy as np
import tensorflow as tf

def preprocess_image(x):
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = vgg19.preprocess_input(x)
    return x

def deprocess_image(x):
    # Mean values of the ImageNet dataset
    x[:,:,0] += 103.939
    x[:,:,1] += 116.779
    x[:,:,2] += 123.68
    # BGR -> RGB
    x = x[:,:,::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x
    
def tf_gram_matrix (x, N, M):
    perm_x = tf.reshape(x,(M,N))
    return tf.matmul(tf.transpose(perm_x), perm_x)

