import tensorflow as tf
import numpy as np

''' learning example (убучающие примеры) '''

train_inp_data = np.array([
    [0.1, 0.1, 0.1, 0.1, 0.1],       # normal mode
    [0, 0.5, 1, 1, 0.5],             # asynchronous mode 1
    [0, 0.2, 0.5, 0.2, 0],           # asynchronous mode 2
    [1, 1, 1, 1, 1],                 # normal mode
    [0.3, 0.3, 1, 0.3, 0.3]          # circuit mode
])

label_train = np.array([
    [0],            # normal mode
    [1],            # asynchronous mode 1
    [1],            # asynchronous mode 2
    [0],            # normal mode
    [0]             # circuit mode
])

''' test (проверка)'''

x_example = np.array([
    [0.5, 0.5, 0.5, 0.5, 0.5],            # normal mode
    [0.1, 0.3, 0.7, 0.3, 0.1],            # asynchronous mode
    [0.3, 0.8, 0.8, 0.3, 0.3]             # circuit mode
])


learning_rate = 0.01    # speed of learning (коэффициент скорости обучения)
epoch = 15000           # epoch's of learning ( количество эпох обучения)
n_input = 5             # neurons in input (число входных нейронов)
n_hidden = 10           # neurons in hidden (число нейронов в скрытом слое)
n_output = 1            # neurons in output (число нейронов в выходе)
pb = 1                  # probability dropout (вероятность dropout)


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

''' other optimze methods '''
#tf.train.AdamOptimizer(learning_rate).minimize(cost)
#tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)

    #print(session.run(W1))

    for step in range(epoch):
        session.run(optimizer, feed_dict={X: train_inp_data, Y: label_train, probability_dropout: pb})
     # print cost:
        if step % 1000 == 0:
            print(session.run(cost, feed_dict={X: train_inp_data, Y: label_train, probability_dropout: pb}))

    answer = tf.equal(tf.floor(out + 0.5), Y)
    accuracy = tf.reduce_mean(tf.cast(answer, "float"))

    print(session.run([out], feed_dict={X: train_inp_data, Y: label_train, probability_dropout: pb}))
    print("Accuracy:", accuracy.eval({X: train_inp_data, Y: label_train, probability_dropout: pb}))

    print(session.run([out], feed_dict = {X: x_example, probability_dropout: pb}))

    #print(session.run(W1))
