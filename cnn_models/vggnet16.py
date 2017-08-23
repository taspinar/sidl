import tensorflow as tf
from utils import *

#The VGGNET-16 Neural Network 
VGG16_FILTER_SIZE_1, VGG16_FILTER_SIZE_2, VGG16_FILTER_SIZE_3, VGG16_FILTER_SIZE_4 = 3, 3, 3, 3
VGG16_FILTER_DEPTH_1, VGG16_FILTER_DEPTH_2, VGG16_FILTER_DEPTH_3, VGG16_FILTER_DEPTH_4 = 64, 128, 256, 512
VGG16_NUM_HIDDEN_1, VGG16_NUM_HIDDEN_2 = 4096, 1000

def variables_vggnet16(filter_size1 = VGG16_FILTER_SIZE_1, filter_size2 = VGG16_FILTER_SIZE_2, 
                       filter_size3 = VGG16_FILTER_SIZE_3, filter_size4 = VGG16_FILTER_SIZE_4, 
                       filter_depth1 = VGG16_FILTER_DEPTH_1, filter_depth2 = VGG16_FILTER_DEPTH_2, 
                       filter_depth3 = VGG16_FILTER_DEPTH_3, filter_depth4 = VGG16_FILTER_DEPTH_4,
                       num_hidden1 = VGG16_NUM_HIDDEN_1, num_hidden2 = VGG16_NUM_HIDDEN_2,
                       image_width = 224, image_height = 224, image_depth = 3, num_labels = 17):
    
    w1 = tf.Variable(tf.truncated_normal([filter_size1, filter_size1, image_depth, filter_depth1], stddev=0.1))
    b1 = tf.Variable(tf.zeros([filter_depth1]))
    w2 = tf.Variable(tf.truncated_normal([filter_size1, filter_size1, filter_depth1, filter_depth1], stddev=0.1))
    b2 = tf.Variable(tf.constant(1.0, shape=[filter_depth1]))

    w3 = tf.Variable(tf.truncated_normal([filter_size2, filter_size2, filter_depth1, filter_depth2], stddev=0.1))
    b3 = tf.Variable(tf.constant(1.0, shape = [filter_depth2]))
    w4 = tf.Variable(tf.truncated_normal([filter_size2, filter_size2, filter_depth2, filter_depth2], stddev=0.1))
    b4 = tf.Variable(tf.constant(1.0, shape = [filter_depth2]))
    
    w5 = tf.Variable(tf.truncated_normal([filter_size3, filter_size3, filter_depth2, filter_depth3], stddev=0.1))
    b5 = tf.Variable(tf.constant(1.0, shape = [filter_depth3]))
    w6 = tf.Variable(tf.truncated_normal([filter_size3, filter_size3, filter_depth3, filter_depth3], stddev=0.1))
    b6 = tf.Variable(tf.constant(1.0, shape = [filter_depth3]))
    w7 = tf.Variable(tf.truncated_normal([filter_size3, filter_size3, filter_depth3, filter_depth3], stddev=0.1))
    b7 = tf.Variable(tf.constant(1.0, shape=[filter_depth3]))

    w8 = tf.Variable(tf.truncated_normal([filter_size4, filter_size4, filter_depth3, filter_depth4], stddev=0.1))
    b8 = tf.Variable(tf.constant(1.0, shape = [filter_depth4]))
    w9 = tf.Variable(tf.truncated_normal([filter_size4, filter_size4, filter_depth4, filter_depth4], stddev=0.1))
    b9 = tf.Variable(tf.constant(1.0, shape = [filter_depth4]))
    w10 = tf.Variable(tf.truncated_normal([filter_size4, filter_size4, filter_depth4, filter_depth4], stddev=0.1))
    b10 = tf.Variable(tf.constant(1.0, shape = [filter_depth4]))
    
    w11 = tf.Variable(tf.truncated_normal([filter_size4, filter_size4, filter_depth4, filter_depth4], stddev=0.1))
    b11 = tf.Variable(tf.constant(1.0, shape = [filter_depth4]))
    w12 = tf.Variable(tf.truncated_normal([filter_size4, filter_size4, filter_depth4, filter_depth4], stddev=0.1))
    b12 = tf.Variable(tf.constant(1.0, shape=[filter_depth4]))
    w13 = tf.Variable(tf.truncated_normal([filter_size4, filter_size4, filter_depth4, filter_depth4], stddev=0.1))
    b13 = tf.Variable(tf.constant(1.0, shape = [filter_depth4]))
    
    no_pooling_layers = 5

    w14 = tf.Variable(tf.truncated_normal([(image_width // (2**no_pooling_layers))*(image_height // (2**no_pooling_layers))*filter_depth4 , num_hidden1], stddev=0.1))
    b14 = tf.Variable(tf.constant(1.0, shape = [num_hidden1]))
    
    w15 = tf.Variable(tf.truncated_normal([num_hidden1, num_hidden2], stddev=0.1))
    b15 = tf.Variable(tf.constant(1.0, shape = [num_hidden2]))
   
    w16 = tf.Variable(tf.truncated_normal([num_hidden2, num_labels], stddev=0.1))
    b16 = tf.Variable(tf.constant(1.0, shape = [num_labels]))
    variables = {
        'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5, 'w6': w6, 'w7': w7, 'w8': w8, 'w9': w9, 'w10': w10, 
        'w11': w11, 'w12': w12, 'w13': w13, 'w14': w14, 'w15': w15, 'w16': w16, 
        'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4, 'b5': b5, 'b6': b6, 'b7': b7, 'b8': b8, 'b9': b9, 'b10': b10, 
        'b11': b11, 'b12': b12, 'b13': b13, 'b14': b14, 'b15': b15, 'b16': b16
    }
    return variables

def model_vggnet16(data, variables):
    layer1_conv = tf.nn.conv2d(data, variables['w1'], [1, 1, 1, 1], padding='SAME')
    layer1_actv = tf.nn.relu(layer1_conv + variables['b1'])
    layer2_conv = tf.nn.conv2d(layer1_actv, variables['w2'], [1, 1, 1, 1], padding='SAME')
    layer2_actv = tf.nn.relu(layer2_conv + variables['b2'])
    layer2_pool = tf.nn.max_pool(layer2_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    layer3_conv = tf.nn.conv2d(layer2_pool, variables['w3'], [1, 1, 1, 1], padding='SAME')
    layer3_actv = tf.nn.relu(layer3_conv + variables['b3'])   
    layer4_conv = tf.nn.conv2d(layer3_actv, variables['w4'], [1, 1, 1, 1], padding='SAME')
    layer4_actv = tf.nn.relu(layer4_conv + variables['b4'])
    layer4_pool = tf.nn.max_pool(layer4_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    layer5_conv = tf.nn.conv2d(layer4_pool, variables['w5'], [1, 1, 1, 1], padding='SAME')
    layer5_actv = tf.nn.relu(layer5_conv + variables['b5'])
    layer6_conv = tf.nn.conv2d(layer5_actv, variables['w6'], [1, 1, 1, 1], padding='SAME')
    layer6_actv = tf.nn.relu(layer6_conv + variables['b6'])
    layer7_conv = tf.nn.conv2d(layer6_actv, variables['w7'], [1, 1, 1, 1], padding='SAME')
    layer7_actv = tf.nn.relu(layer7_conv + variables['b7'])
    layer7_pool = tf.nn.max_pool(layer7_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    layer8_conv = tf.nn.conv2d(layer7_pool, variables['w8'], [1, 1, 1, 1], padding='SAME')
    layer8_actv = tf.nn.relu(layer8_conv + variables['b8'])
    layer9_conv = tf.nn.conv2d(layer8_actv, variables['w9'], [1, 1, 1, 1], padding='SAME')
    layer9_actv = tf.nn.relu(layer9_conv + variables['b9'])
    layer10_conv = tf.nn.conv2d(layer9_actv, variables['w10'], [1, 1, 1, 1], padding='SAME')
    layer10_actv = tf.nn.relu(layer10_conv + variables['b10'])
    layer10_pool = tf.nn.max_pool(layer10_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    layer11_conv = tf.nn.conv2d(layer10_pool, variables['w11'], [1, 1, 1, 1], padding='SAME')
    layer11_actv = tf.nn.relu(layer11_conv + variables['b11'])
    layer12_conv = tf.nn.conv2d(layer11_actv, variables['w12'], [1, 1, 1, 1], padding='SAME')
    layer12_actv = tf.nn.relu(layer12_conv + variables['b12'])
    layer13_conv = tf.nn.conv2d(layer12_actv, variables['w13'], [1, 1, 1, 1], padding='SAME')
    layer13_actv = tf.nn.relu(layer13_conv + variables['b13'])
    layer13_pool = tf.nn.max_pool(layer13_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    
    flat_layer  = flatten_tf_array(layer13_pool)
    layer14_fccd = tf.matmul(flat_layer, variables['w14']) + variables['b14']
    layer14_actv = tf.nn.relu(layer14_fccd)
    layer14_drop = tf.nn.dropout(layer14_actv, 0.5)
    
    layer15_fccd = tf.matmul(layer14_drop, variables['w15']) + variables['b15']
    layer15_actv = tf.nn.relu(layer15_fccd)
    layer15_drop = tf.nn.dropout(layer15_actv, 0.5)
    
    logits = tf.matmul(layer15_drop, variables['w16']) + variables['b16']
    return logits