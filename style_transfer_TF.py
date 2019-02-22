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

# Global parameters
style_layers = ['vgg19/conv1/1/Relu:0', 'vgg19/conv2/1/Relu:0', 'vgg19/conv3/1/Relu:0', 'vgg19/conv4/1/Relu:0', 'vgg19/conv5/1/Relu:0']
style_weights = [4,                    3,                    1.5,                    1,                    0.5           ]
content_layers = ['vgg19/conv4/2/Relu:0']
alpha_list = [1e-1]
beta = 5.0

############### Model build ###############################
# build the VGG19 network with ImageNet weights, no fully connected layers
inputs = tf.placeholder(tf.float32, [None, 224, 224, 3])
model = nets.VGG19(inputs)
outputs_dict = dict([layer.name, layer] for layer in model.get_outputs())
#exit()

# Style Image
img = nets.utils.load_img('vangogh.jpg', target_size=(img_width, img_height))
if (show_pictures == 1):
    imshow(img[0])
imsave("original_style.png", img[0])
style_img = model.preprocess(img)

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

for content_layer in content_layers:

    for alpha in alpha_list:

        print('Content layer =', content_layer)
        print('Alpha =', str(alpha))

        ## Style loss
        style_loss = 0
        i = 0
        for style_layer in style_layers:
            layer = outputs_dict[style_layer]
            N = layer.shape[3]
            M = layer.shape[1] * layer.shape[2]
            style_feature = sess.run (layer, feed_dict = {inputs: style_img})
            style_feature_tensor = tf.constant (style_feature)
            S = tf_gram_matrix(style_feature_tensor, N, M)
            L = tf_gram_matrix(layer, N, M)
            l = tf.reduce_sum(tf.pow(S - L, 2))
            l /= 4*(N.value**2)*(M.value**2)
            style_loss += l * style_weights[i]
            i += 1
        
        ## Content loss
        layer = outputs_dict[content_layer]
        content_feature = sess.run (layer, feed_dict = {inputs: content_img})
        content_feature_tensor = tf.constant (content_feature)
        content_loss = tf.reduce_sum(tf.pow(content_feature_tensor - layer, 2))
        
        ## Total loss
        loss = alpha * content_loss + beta * style_loss
        
        gradients = tf.gradients (loss, inputs)
        
        # Gradients for white noise image using min_l_bfgs_b 
        def fn_loss (x):
            x = x.reshape((1, img_width, img_height, 3))
            ll = np.array (sess.run (loss, feed_dict = {inputs: x}))
            l = ll.sum()
            return l.flatten().astype('float64')
            
        def fn_grads (x):
            x = x.reshape((1, img_width, img_height, 3))
            g = np.array (sess.run (gradients, feed_dict = {inputs: x}))[0]
            return g.flatten().astype('float64')
        
        # we start from a white noise image with some random noise
        gray_img = np.random.random((1, img_width, img_height, 3)) # Channel Last
        
        for i in range (0, bfgs_iter):
            print ('iteration ',i)
            gray_img, min_val, info = fmin_l_bfgs_b(fn_loss, gray_img, fn_grads, maxfun=20)
            print('Current loss value:', min_val.sum())
                    
        rec_img = gray_img.reshape((img_width, img_height, 3))
        bfgs='bfgs_'
        
        
        # decode the resulting input image
        reconstructed_img = deprocess_image(rec_img)
        if (show_pictures == 1):
            imshow(reconstructed_img)
        img_name = 'result'+bfgs+content_layer+'__alpha='+str(alpha)+'.png'
        img_name = img_name.replace ("/", "_")
        imsave(img_name, reconstructed_img)
