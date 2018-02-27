import tensorflow as tf
import numpy as np

x_data = np.array([
    [0, 0, 0, 0, 0],
    [0, 0.5, 1, 0.5, 0],
    [0, 0.2, 0.5, 0.2, 0],
    [1, 1, 1, 1, 1]
])

y_data = np.array([
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

W1 = tf.Variable(tf.random_uniform([n_input, n_hidden], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([n_hidden, n_output], -1.0, 1.0))

b1 = tf.Variable(tf.ones([n_hidden], name="Bias_hiden"))
b2 = tf.Variable(tf.ones([n_output], name="Bias_output"))

L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
out = tf.sigmoid(tf.matmul(L2, W2) + b2)

cost = tf.reduce_mean(-Y * tf.log(out) - (1 - Y) * tf.log(1 - out)) # croosentropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)

    #print(session.run(W1))

    for step in range(epoch):
        session.run(optimizer, feed_dict={X: x_data, Y: y_data})

        if step % 1000 == 0:
            print(session.run(cost, feed_dict={X: x_data, Y: y_data}))

    answer = tf.equal(tf.floor(out + 0.5), Y)
    accuracy = tf.reduce_mean(tf.cast(answer, "float"))

    print(session.run([out], feed_dict={X: x_data, Y: y_data}))
    print("Accuracy:", accuracy.eval({X: x_data, Y: y_data}))

    print(session.run([out], feed_dict = {X: x_example}))

    #print(session.run(W1))