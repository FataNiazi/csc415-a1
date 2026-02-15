from tensorflow.python.layers import base
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from functools import partial

# TODO: make activation parameterized in all these functions

def ssam(conv_layer):
    """spatial soft argmax"""
    _, num_rows, num_cols, num_fp = conv_layer.get_shape()
    num_rows, num_cols, num_fp = [int(x) for x in [num_rows, num_cols, num_fp]]
    x_map = np.empty([num_rows, num_cols], np.float32)
    y_map = np.empty([num_rows, num_cols], np.float32)
    
    for i in range(num_rows):
        for j in range(num_cols):
            x_map[i, j] = (i - num_rows / 2.0) / num_rows
            y_map[i, j] = (j - num_cols / 2.0) / num_cols
    
    x_map = tf.convert_to_tensor(x_map)
    y_map = tf.convert_to_tensor(y_map)
    
    x_map = tf.reshape(x_map, [num_rows * num_cols])
    y_map = tf.reshape(y_map, [num_rows * num_cols])
    
    # rearrange features to be [batch_size, num_fp, num_rows, num_cols]
    features = tf.reshape(tf.transpose(conv_layer, [0,3,1,2]),
                          [-1, num_rows*num_cols])
    softmax = tf.nn.softmax(features)
    
    fp_x = tf.reduce_sum(tf.multiply(x_map, softmax), [1], keep_dims=True)
    fp_y = tf.reduce_sum(tf.multiply(y_map, softmax), [1], keep_dims=True)
    
    conv_out_flat = tf.reshape(tf.concat(axis=1, values=[fp_x, fp_y]), [-1, num_fp*2])
    return conv_out_flat

def bn_conv(inputs, **kwargs):
    """Batch norm convolution layer"""
    out = tf.keras.layers.conv2d(inputs, **kwargs)
    out = tf.keras.layers.BatchNormalization(fused=True)(out, training=True)
    out = tf.nn.relu(out)
    return out

def flatten(inputs, use_ssam):
    """Flatten conv out using reshape or spatial soft argmax"""
    if use_ssam:
        out = ssam(inputs)
        #out = tf.contrib.layers.spatial_softmax(inputs, name='ssam')
    else:
        out = tf.keras.layers.Flatten()(inputs)
    return out 

def binned_head(inputs, outdim, hparams):
    """Output of binned network, make heads for xyz and stack them"""
    outx = tf.keras.layers.Dense(outdim, activation=None)(inputs)
    outy = tf.keras.layers.Dense(outdim, activation=None)(inputs)
    outz = tf.keras.layers.Dense(outdim, activation=None)(inputs)

    out = tf.stack([outx, outy, outz], 1)
    return out

def trivial_forward(inputs, outdim, hparams):
    """for testing speed"""
    out = tf.contrib.layers.spatial_softmax(inputs)
    out = tf.keras.layers.Dense(outdim)(out)
    return out

# TODO: convert all the extra args to a dictionary
#s2_layers=None, s1_layers=None, fc_layers=None, batch_norm=False, ssam=False
def vgg_forward(inputs, outdim, hparams):
    """Forward pass of VGG network"""
    def maxpool(x):
        return tf.nn.max_pool2d(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv2d = lambda inputs, filters: tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)
    vgg_bn_conv = partial(bn_conv, kernel_size=3, strides=(1,1), padding='SAME', activation=tf.nn.relu)
    conv = vgg_bn_conv if hparams['batch_norm'] else conv2d # add batch norm

    out = inputs

    out = conv(inputs=out, filters=64)
    out = conv(inputs=out, filters=64)
    out = maxpool(out)

    out = conv(inputs=out, filters=128)
    out = conv(inputs=out, filters=128)
    out = maxpool(out)

    out = conv(inputs=out, filters=256)
    out = conv(inputs=out, filters=256)
    out = conv(inputs=out, filters=256)
    out = maxpool(out)

    out = conv(inputs=out, filters=512)
    out = conv(inputs=out, filters=512)
    out = conv(inputs=out, filters=512)
    out = maxpool(out)

    out = conv(inputs=out, filters=512)
    out = conv(inputs=out, filters=512)
    out = conv(inputs=out, filters=512)
    out = maxpool(out)

    out = flatten(out, hparams['ssam'])

    out = tf.keras.layers.Dense(256, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dense(64, activation=tf.nn.relu)(out)

    if hparams['output'] == 'xyz': 
        out = tf.keras.layers.Dense(outdim, kernel_initializer=None)(out)
    elif hparams['output'] == 'binned':
        out = binned_head(out, outdim, hparams)
    return out

def reg_forward(inputs, outdim, hparams):
    """
    Forward pass of neural network

    Parameterized versions of:
    input
    3x3 with stride 2
    3x3 with stirde 1
    flatten
    fc
    out
    """
    conv2d = partial(tf.keras.layers.conv2d, kernel_size=3, padding='SAME', activation=tf.nn.relu) # uses 4x3, stride 1, zero-padding throughout
    bn_conv2d = partial(bn_conv, kernel_size=3, padding='SAME', activation=tf.nn.relu)
    conv = bn_conv2d if hparams['batch_norm ']else conv2d # add batch norm

    out = inputs

    for s2_layer in hparams['s2_layers']:
        out = conv(out, s2_layer, 3, strides=(2,2), padding='SAME')
    for s1_layer in hparams['s1_layers']:
        out = conv(out, s1_layer, 3, strides=(1,1), padding='SAME')

    out = flatten(out, hparams['ssam'])

    for fc_layer in hparams['fc_layers']:
        out = tf.keras.layers.Dense(fc_layer, activation=tf.nn.relu)(out)

    out = tf.keras.layers.Dense(outdim)(out)
    return out
