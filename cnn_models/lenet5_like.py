import tensorflow as tf
from utils import *

LENET5_LIKE_BATCH_SIZE = 32
LENET5_LIKE_FILTER_SIZE = 5
LENET5_LIKE_FILTER_DEPTH = 16
LENET5_LIKE_NUM_HIDDEN = 120

def variables_lenet5_like(filter_size = LENET5_LIKE_FILTER_SIZE, 
                          filter_depth = LENET5_LIKE_FILTER_DEPTH, 
                          num_hidden = LENET5_LIKE_NUM_HIDDEN,
                          image_width = 28, image_height = 28, image_depth = 1, num_labels = 10):
    
    w1 = tf.Variable(tf.truncated_normal([filter_size, filter_size, image_depth, filter_depth], stddev=0.1))
    b1 = tf.Variable(tf.zeros([filter_depth]))

    w2 = tf.Variable(tf.truncated_normal([filter_size, filter_size, filter_depth, filter_depth], stddev=0.1))
    b2 = tf.Variable(tf.constant(1.0, shape=[filter_depth]))
   
    w3 = tf.Variable(tf.truncated_normal([(image_width // 4)*(image_height // 4)*filter_depth , num_hidden], stddev=0.1))
    b3 = tf.Variable(tf.constant(1.0, shape = [num_hidden]))

    w4 = tf.Variable(tf.truncated_normal([num_hidden, num_hidden], stddev=0.1))
    b4 = tf.Variable(tf.constant(1.0, shape = [num_hidden]))
    
    w5 = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    b5 = tf.Variable(tf.constant(1.0, shape = [num_labels]))
    variables = {
        'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5,
        'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4, 'b5': b5
    }
    return variables

def model_lenet5_like(data, variables):
    layer1_conv = tf.nn.conv2d(data, variables['w1'], [1, 1, 1, 1], padding='SAME')
    layer1_actv = tf.nn.relu(layer1_conv + variables['b1'])
    layer1_pool = tf.nn.avg_pool(layer1_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    layer2_conv = tf.nn.conv2d(layer1_pool, variables['w2'], [1, 1, 1, 1], padding='SAME')
    layer2_actv = tf.nn.relu(layer2_conv + variables['b2'])
    layer2_pool = tf.nn.avg_pool(layer2_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    
    flat_layer = flatten_tf_array(layer2_pool)
    layer3_fccd = tf.matmul(flat_layer, variables['w3']) + variables['b3']
    layer3_actv = tf.nn.relu(layer3_fccd)
    #layer3_drop = tf.nn.dropout(layer3_actv, 0.5)
    
    layer4_fccd = tf.matmul(layer3_actv, variables['w4']) + variables['b4']
    layer4_actv = tf.nn.relu(layer4_fccd)
    #layer4_drop = tf.nn.dropout(layer4_actv, 0.5)
    
    logits = tf.matmul(layer4_actv, variables['w5']) + variables['b5']
    return logits