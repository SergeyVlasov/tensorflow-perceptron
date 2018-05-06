import tensorflow as tf
import numpy as np

train_inp_data = np.array([
    [0.1, 0.1, 0.1, 0.1, 0.1],
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


learning_rate = 0.01
epoch = 10000
n_input = 5
n_hidden = 10
n_output = 1

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

Weights_inp_to_hid = tf.Variable(tf.random_uniform([n_input, n_hidden], -1.0, 1.0))
Weights_hid_to_out = tf.Variable(tf.random_uniform([n_hidden, n_output], -1.0, 1.0))

bias_hid = tf.Variable(tf.ones([n_hidden], name="Bias_hiden"))
bias_out = tf.Variable(tf.ones([n_output], name="Bias_output"))

L2 = tf.sigmoid(tf.matmul(X, Weights_inp_to_hid) + bias_hid)
probability_dropout = tf.placeholder(tf.float32)
L2 = tf.nn.dropout(L2, probability_dropout)


out = tf.sigmoid(tf.matmul(L2, Weights_hid_to_out) + bias_out)


cost = tf.reduce_mean(-Y * tf.log(out) - (1 - Y) * tf.log(1 - out)) # croosentropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#tf.train.AdamOptimizer(learning_rate).minimize(cost)
#tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)

    #print(session.run(W1))

    for step in range(epoch):
        session.run(optimizer, feed_dict={X: train_inp_data, Y: label_train, probability_dropout: 1})
     # print cost:
        if step % 1000 == 0:
            print(session.run(cost, feed_dict={X: train_inp_data, Y: label_train, probability_dropout: 1}))

    answer = tf.equal(tf.floor(out + 0.5), Y)
    accuracy = tf.reduce_mean(tf.cast(answer, "float"))

    print(session.run([out], feed_dict={X: train_inp_data, Y: label_train, probability_dropout: 1}))
    print("Accuracy:", accuracy.eval({X: train_inp_data, Y: label_train, probability_dropout: 1}))

    print(session.run([out], feed_dict = {X: x_example, probability_dropout: 1}))

    #print(session.run(W1))
