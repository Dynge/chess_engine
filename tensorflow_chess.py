import tensorflow as tf

learning_rate = 0.001
training_iters = 10
batch_size = 32
n_input = 8
n_classes = 64
board_state = tf.placeholder("float", [None, n_input, n_input, 1])
moving_from = tf.placeholder("float", [None, n_classes])


def conv2d(x, W, b, strides=1):
    """
    Convolutional 2D Layer. Uses RELU activation and has bias and weights.
    """
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding="SAME")
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


layer_sizes = {"conv1": 32, "conv2": 64, "flatten": 128, "out": 64}

weights = {
    "wc1": tf.get_variable(
        "W0",
        shape=(3, 3, 1, layer_sizes["conv1"]),
        initializer=tf.contrib.layers.xavier_initializer(),
    ),
    "wc2": tf.get_variable(
        "W1",
        shape=(3, 3, 1, layer_sizes["conv2"]),
        initializer=tf.contrib.layers.xavier_initializer(),
    ),
    "flatten": tf.get_variable(
        "Wflat",
        shape=(8 * 8 * layer_sizes["conv2"], layer_sizes["flatten"]),
        initializer=tf.contrib.layers.xavier_initializer(),
    ),
    "out": tf.get_variable(
        "Wout",
        shape=(layer_sizes["out"], n_classes),
        initializer=tf.contrib.layers.xavier_initializer(),
    ),
}

biases = {
    "bc1": tf.get_variable(
        "B0",
        shape=(layer_sizes["conv1"]),
        initializer=tf.contrib.layers.xavier_initializer(),
    ),
    "bc2": tf.get_variable(
        "B1",
        shape=(layer_sizes["conv2"]),
        initializer=tf.contrib.layers.xavier_initializer(),
    ),
    "flatten": tf.get_variable(
        "Bflat",
        shape=(layer_sizes["flatten"]),
        initializer=tf.contrib.layers.xavier_initializer(),
    ),
    "out": tf.get_variable(
        "Bout", shape=(n_classes), initializer=tf.contrib.layers.xavier_initializer()
    ),
}


def conv_net(x, weights, biases):
    """
    The full network architecture of the convolutional network.
    """
    conv1 = conv2d(x, weights["wc1"], biases["bc1"])
    conv2 = conv2d(conv1, weights["wc2"], biases["bc2"])
    fc1 = tf.reshape(conv2, [-1, weights["flatten"].getshape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights["flatten"]), biases["flatten"])
    fc1 = tf.nn.relu(fc1)

    out = tf.add(tf.matmul(fc1, weights["out"]), biases["out"])
    return out


pred = conv_net(board_state, weights, biases)
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=moving_from)
)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(moving_from, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

total_X = np.load("./data/game_positions.npy")
total_y = np.load("./data/move_positions.npy")

split_point = int(len(total_X) * 0.8)
train_X = total_X[0:split_point]
test_X = total_X[split_point : len(total_X)]
train_y = total_y[0:split_point]
test_y = total_y[split_point : len(total_X)]


with tf.Session() as sess:
    sess.run(init)
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter("./output", sess.graph)
    for i in range(training_iters):
        for batch in range(len(train_X)):
            batch_x = train_X[
                batch * batch_size : min((batch + 1) * batch_size, len(train_X))
            ]
            batch_y = train_y[
                batch * batch_size : min((batch + 1) * batch_size, len(train_y))
            ]

            opt = sess.run(optimizer, feed_dict={"x": batch_x, "y": batch_y})
            loss, acc = sess.run(
                [cost, accuracy], feed_dict={"x": batch_x, "y": batch_y}
            )
        print(
            "Iter "
            + str(i)
            + ", Loss= "
            + "{:.6f}".format(loss)
            + ", Training Accuracy= "
            + "{:.5f}".format(acc)
        )
        print("Optimization Finished!")

        # Calculate accuracy for all 10000 mnist test images
        test_acc, valid_loss = sess.run(
            [accuracy, cost], feed_dict={"x": test_X, "y": test_y}
        )
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print("Testing Accuracy:", "{:.5f}".format(test_acc))
    summary_writer.close()
