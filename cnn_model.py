import tensorflow as tf
from cnn_models import lenet5, lenet5_like, alexnet, vggnet16
from utils import *

#Variables used in the constructing and running the graph
num_steps = 10001
display_step = 1000
#learning_rate = 0.5
batch_size = 32

#General
image_width = ox17_image_width
image_height = ox17_image_height
image_depth = ox17_image_depth
num_labels = ox17_num_labels

test_dataset = test_dataset_ox17
test_labels = test_labels_ox17
train_dataset = train_dataset_ox17
train_labels = train_labels_ox17

graph = tf.Graph()
with graph.as_default():
    #1) First we put the input data in a tensorflow friendly form. 
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_width, image_height, image_depth))
    tf_train_labels = tf.placeholder(tf.float32, shape = (batch_size, num_labels))
    tf_test_dataset = tf.constant(test_dataset, tf.float32)

    #2) Then, the weight matrices and bias vectors are initialized
    variables_ = variables_lenet5()
    #variables = variables_lenet5_like()
    #variables = variables_alexnet()
    #variables = variables_vggnet16()
    variables = variables_(image_width = image_width, image_height=image_height, image_depth = image_depth, num_labels = num_labels)


    #3. The model used to calculate the logits (predicted labels)
    model = model_lenet5
    #model = model_lenet5_like
    #model = model_alexnet
    #model = model_vggnet16
    logits = model(tf_train_dataset, variables)

    #4. then we compute the softmax cross entropy between the logits and the (actual) labels
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))

    #5. The optimizer is used to calculate the gradients of the loss function 
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    #optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.0).minimize(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(model(tf_test_dataset, variables))


with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized with learning_rate', learning_rate)
    for step in range(num_steps):
        #Since we are using stochastic gradient descent, we are selecting  small batches from the training dataset,
        #and training the convolutional neural network each time with a batch. 
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        train_accuracy = accuracy(predictions, batch_labels)
        
        if step % display_step == 0:
            test_accuracy = accuracy(test_prediction.eval(), test_labels)
            message = "step {:04d} : loss is {:06.2f}, accuracy on training set {:02.2f} %, accuracy on test set {:02.2f} %".format(step, l, train_accuracy, test_accuracy)
            print(message)