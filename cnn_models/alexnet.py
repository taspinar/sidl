import tensorflow as tf
from utils import *

ALEX_FILTER_DEPTH_1, ALEX_FILTER_DEPTH_2, ALEX_FILTER_DEPTH_3, ALEX_FILTER_DEPTH_4 = 96, 256, 384, 256
ALEX_FILTER_SIZE_1, ALEX_FILTER_SIZE_2, ALEX_FILTER_SIZE_3, ALEX_FILTER_SIZE_4 = 11, 5, 3, 3
ALEX_NUM_HIDDEN_1, ALEX_NUM_HIDDEN_2 = 4096, 4096

def variables_alexnet(filter_size1 = ALEX_FILTER_SIZE_1, filter_size2 = ALEX_FILTER_SIZE_2, 
                      filter_size3 = ALEX_FILTER_SIZE_3, filter_size4 = ALEX_FILTER_SIZE_4, 
                      filter_depth1 = ALEX_FILTER_DEPTH_1, filter_depth2 = ALEX_FILTER_DEPTH_2, 
                      filter_depth3 = ALEX_FILTER_DEPTH_3, filter_depth4 = ALEX_FILTER_DEPTH_4, 
                      num_hidden1 = ALEX_NUM_HIDDEN_1, num_hidden2 = ALEX_NUM_HIDDEN_2,
                      image_width = 224, image_height = 224, image_depth = 3, num_labels = 17):
    
    w1 = tf.Variable(tf.truncated_normal([filter_size1, filter_size1, image_depth, filter_depth1], stddev=0.1))
    b1 = tf.Variable(tf.zeros([filter_depth1]))

    w2 = tf.Variable(tf.truncated_normal([filter_size2, filter_size2, filter_depth1, filter_depth2], stddev=0.1))
    b2 = tf.Variable(tf.constant(1.0, shape=[filter_depth2]))

    w3 = tf.Variable(tf.truncated_normal([filter_size3, filter_size3, filter_depth2, filter_depth3], stddev=0.1))
    b3 = tf.Variable(tf.zeros([filter_depth3]))

    w4 = tf.Variable(tf.truncated_normal([filter_size4, filter_size4, filter_depth3, filter_depth3], stddev=0.1))
    b4 = tf.Variable(tf.constant(1.0, shape=[filter_depth3]))
    
    w5 = tf.Variable(tf.truncated_normal([filter_size4, filter_size4, filter_depth3, filter_depth3], stddev=0.1))
    b5 = tf.Variable(tf.zeros([filter_depth3]))
       
    pool_reductions = 3
    conv_reductions = 2
    no_reductions = pool_reductions + conv_reductions
    w6 = tf.Variable(tf.truncated_normal([(image_width // 2**no_reductions)*(image_height // 2**no_reductions)*filter_depth3, num_hidden1], stddev=0.1))
    b6 = tf.Variable(tf.constant(1.0, shape = [num_hidden1]))

    w7 = tf.Variable(tf.truncated_normal([num_hidden1, num_hidden2], stddev=0.1))
    b7 = tf.Variable(tf.constant(1.0, shape = [num_hidden2]))
    
    w8 = tf.Variable(tf.truncated_normal([num_hidden2, num_labels], stddev=0.1))
    b8 = tf.Variable(tf.constant(1.0, shape = [num_labels]))
    
    variables = {
        'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5, 'w6': w6, 'w7': w7, 'w8': w8, 
        'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4, 'b5': b5, 'b6': b6, 'b7': b7, 'b8': b8
    }
    return variables


def model_alexnet(data, variables):
    layer1_conv = tf.nn.conv2d(data, variables['w1'], [1, 4, 4, 1], padding='SAME')
    layer1_relu = tf.nn.relu(layer1_conv + variables['b1'])
    layer1_pool = tf.nn.max_pool(layer1_relu, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
    layer1_norm = tf.nn.local_response_normalization(layer1_pool)
    
    layer2_conv = tf.nn.conv2d(layer1_norm, variables['w2'], [1, 1, 1, 1], padding='SAME')
    layer2_relu = tf.nn.relu(layer2_conv + variables['b2'])
    layer2_pool = tf.nn.max_pool(layer2_relu, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
    layer2_norm = tf.nn.local_response_normalization(layer2_pool)
    
    layer3_conv = tf.nn.conv2d(layer2_norm, variables['w3'], [1, 1, 1, 1], padding='SAME')
    layer3_relu = tf.nn.relu(layer3_conv + variables['b3'])
    
    layer4_conv = tf.nn.conv2d(layer3_relu, variables['w4'], [1, 1, 1, 1], padding='SAME')
    layer4_relu = tf.nn.relu(layer4_conv + variables['b4'])
    
    layer5_conv = tf.nn.conv2d(layer4_relu, variables['w5'], [1, 1, 1, 1], padding='SAME')
    layer5_relu = tf.nn.relu(layer5_conv + variables['b5'])
    layer5_pool = tf.nn.max_pool(layer4_relu, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
    layer5_norm = tf.nn.local_response_normalization(layer5_pool)
    
    flat_layer = flatten_tf_array(layer5_norm)
    layer6_fccd = tf.matmul(flat_layer, variables['w6']) + variables['b6']
    layer6_tanh = tf.tanh(layer6_fccd)
    layer6_drop = tf.nn.dropout(layer6_tanh, 0.5)
    
    layer7_fccd = tf.matmul(layer6_drop, variables['w7']) + variables['b7']
    layer7_tanh = tf.tanh(layer7_fccd)
    layer7_drop = tf.nn.dropout(layer7_tanh, 0.5)
    
    logits = tf.matmul(layer7_drop, variables['w8']) + variables['b8']
    return logits