import tensorflow as tf
from utils import *

LENET5_BATCH_SIZE = 32
LENET5_FILTER_SIZE = 5
LENET5_FILTER_DEPTH_1 = 6
LENET5_FILTER_DEPTH_2 = 16
LENET5_NUM_HIDDEN_1 = 120
LENET5_NUM_HIDDEN_2 = 84

def variables_lenet5(filter_size = LENET5_FILTER_SIZE, filter_depth1 = LENET5_FILTER_DEPTH_1, 
                     filter_depth2 = LENET5_FILTER_DEPTH_2, 
                     num_hidden1 = LENET5_NUM_HIDDEN_1, num_hidden2 = LENET5_NUM_HIDDEN_2,
                     image_width = 28, image_height = 28, image_depth = 1, num_labels = 10):
    
    w1 = tf.Variable(tf.truncated_normal([filter_size, filter_size, image_depth, filter_depth1], stddev=0.1))
    b1 = tf.Variable(tf.zeros([filter_depth1]))

    w2 = tf.Variable(tf.truncated_normal([filter_size, filter_size, filter_depth1, filter_depth2], stddev=0.1))
    b2 = tf.Variable(tf.constant(1.0, shape=[filter_depth2]))

    w3 = tf.Variable(tf.truncated_normal([(image_width // 5)*(image_height // 5)*filter_depth2, num_hidden1], stddev=0.1))
    b3 = tf.Variable(tf.constant(1.0, shape = [num_hidden1]))

    w4 = tf.Variable(tf.truncated_normal([num_hidden1, num_hidden2], stddev=0.1))
    b4 = tf.Variable(tf.constant(1.0, shape = [num_hidden2]))
    
    w5 = tf.Variable(tf.truncated_normal([num_hidden2, num_labels], stddev=0.1))
    b5 = tf.Variable(tf.constant(1.0, shape = [num_labels]))
    variables = {
        'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5,
        'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4, 'b5': b5
    }
    return variables

def model_lenet5(data, variables):
    layer1_conv = tf.nn.conv2d(data, variables['w1'], [1, 1, 1, 1], padding='SAME')
    layer1_actv = tf.sigmoid(layer1_conv + variables['b1'])
    layer1_pool = tf.nn.avg_pool(layer1_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    layer2_conv = tf.nn.conv2d(layer1_pool, variables['w2'], [1, 1, 1, 1], padding='VALID')
    layer2_actv = tf.sigmoid(layer2_conv + variables['b2'])
    layer2_pool = tf.nn.avg_pool(layer2_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    flat_layer = flatten_tf_array(layer2_pool)
    layer3_fccd = tf.matmul(flat_layer, variables['w3']) + variables['b3']
    layer3_actv = tf.nn.sigmoid(layer3_fccd)
    
    layer4_fccd = tf.matmul(layer3_actv, variables['w4']) + variables['b4']
    layer4_actv = tf.nn.sigmoid(layer4_fccd)
    logits = tf.matmul(layer4_actv, variables['w5']) + variables['b5']
    return logits