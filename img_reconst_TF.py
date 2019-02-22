# RCP209 Project - Implementation of Image Style Transfer
# Gilles Augustins - 22/02/2019

from scipy.misc import imsave, imshow
import numpy as np
import tensorflow as tf
import tensornets as nets
from utils import *
from scipy.optimize import fmin_l_bfgs_b

# dimensions of the generated pictures for each filter.
img_width, img_height = 224, 224 
show_pictures = 1

# Optimization algo
# Turn BFGS to 0 for using gradient descent algo 
BFGS = 1
# Parameters of BFGS 
bfgs_iter = 5
# Parameters of the gradient descent
gdsc_step = 0.00001
gdsc_iter = 50

############### Model build ###############################
# build the VGG19 network with ImageNet weights, no fully connected layers
inputs = tf.placeholder(tf.float32, [None, 224, 224, 3])
model = nets.VGG19(inputs)
outputs_dict = dict([layer.name, layer] for layer in model.get_outputs())
layer_name = 'vgg19/conv4/2/Relu:0'

# Content Image
img = nets.utils.load_img('elephant.jpg', target_size=(img_width, img_height))
if (show_pictures == 1):
    imshow(img[0])
imsave("original_content.png", img[0])
content_img = model.preprocess(img)

############### Response from Content image  ###############################

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
sess.run (model.pretrained())

layer = outputs_dict[layer_name]
content_feature = sess.run (layer, feed_dict = {inputs: content_img})
content_feature_tensor = tf.constant (content_feature)

# we start from a white noise image with some random noise
gray_img = np.random.random((1, img_width, img_height, 3)) # Channel Last

loss = tf.math.square(content_feature_tensor - layer)
gradients = tf.gradients (loss, inputs)

if BFGS == 1:
    # Gradients for white noise image using min_l_bfgs_b 
    def fn_loss (x):
        x = x.reshape((1, img_width, img_height, 3))
        ll = np.array (sess.run (loss, feed_dict = {inputs: x}))[0]
        l = ll.sum()
        return l.flatten().astype('float64')
    
    def fn_grads (x):
        x = x.reshape((1, img_width, img_height, 3))
        g = np.array (sess.run (gradients, feed_dict = {inputs: x}))[0]
        return g.flatten().astype('float64')

    for i in range (0, bfgs_iter):
        print ('iteration ',i)
        gray_img, min_val, info = fmin_l_bfgs_b(fn_loss, gray_img, fn_grads, maxfun=20)
        print('Current loss value:', min_val.sum())
            
    rec_img = gray_img.reshape((img_width, img_height, 3))
    bfgs='bfgs_'

else: 
    for i in range (gdsc_iter):
        x = sess.run ([gradients, loss], feed_dict = {inputs: gray_img})
        grads_value = np.array (x[0])[0]
        loss_value = np.array (x[1])
        gray_img -= grads_value * gdsc_step
        print('[', layer_name, ']', 'Iter', i, ': Current loss value =', loss_value.sum())
    bfgs=''
    rec_img = gray_img[0]

# decode the resulting input image
reconstructed_img = deprocess_image(rec_img)
if (show_pictures == 1):
    imshow(reconstructed_img)
img_name = 'img_reconst_'+bfgs+layer_name+'.png'
#imsave(img_name, reconstructed_img)
