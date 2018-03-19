import tensorflow as tf
import numpy as np

train_inp_data = np.array([
    [0, 0, 0, 0, 0],
    [0, 0.5, 1, 0.5, 0],
    [0, 0.2, 0.5, 0.2, 0],
    [1, 1, 1, 1, 1]
])

label_train = np.array([
    [0],
    [1],
    [1],
    [0]
])


x_example = np.array([
    [0.5, 0.5, 0.5, 0.5, 0.5],
    [0.1, 0.3, 0.7, 0.3, 0.1]
])


learning_rate = 0.1
epoch = 5000
n_input = 5
n_hidden = 10
n_output = 1

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

Weights_inp_to_hid = tf.Variable(tf.random_uniform([n_input, n_hidden], -1.0, 1.0))
Weights_hid_to_out = tf.Variable(tf.random_uniform([n_hidden, n_output], -1.0, 1.0))

bias_hid = tf.Variable(tf.ones([n_hidden], name="Bias_hiden"))
bias_out = tf.Variable(tf.ones([n_output], name="Bias_output"))

layer_hid = tf.sigmoid(tf.matmul(X, Weights_inp_to_hid) + bias_hid)
out = tf.sigmoid(tf.matmul(layer_hid, Weights_hid_to_out) + bias_out)

cost_func = tf.losses.log_loss(Y, out) # croosentropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_func)

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
     
    #weights before training    
    #print(session.run(Weights_inp_to_hid))

    for step in range(epoch):
        session.run(optimizer, feed_dict={X: train_inp_data, Y: label_train})
     # print cost
        if step % 1000 == 0:
            print(session.run(cost_func, feed_dict={X: train_inp_data, Y: label_train}))

    answer = tf.equal(tf.floor(out + 0.5), Y)
    accuracy = tf.reduce_mean(tf.cast(answer, "float"))

    print(session.run([out], feed_dict={X: train_inp_data, Y: label_train}))
    print("Accuracy:", accuracy.eval({X: train_inp_data, Y: label_train}))

    print(session.run([out], feed_dict = {X: x_example}))
    
    #weights after training 
    #print(session.run(Weights_inp_to_hid))
