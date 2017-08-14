def variables_fccd(image_width = 28, image_height = 28, num_labels = 10, image_depth = 1):
    weights = tf.Variable(tf.truncated_normal([image_width * image_height*image_depth, num_labels]), tf.float32)
    bias = tf.Variable(tf.zeros([num_labels]), tf.float32)
    variables = { 'w': weights, 'b': bias}
    return variables


def model_fccd(data, variables):
    return tf.matmul(flatten_tf_array(data), variables['w']) + variables['b']