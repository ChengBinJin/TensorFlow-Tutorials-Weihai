# Design the two-layer neural network to solve XOR problem.
import tensorflow as tf

x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [[0], [1], [1], [0]]

# Define placeholders X and Y
X = tf.compat.v1.placeholder(tf.float32, shape=[None, 2], name='X')
Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='Y')

# Define variables W1, W2, b1, and b2
W1 = tf.Variable(tf.random.normal([2, 2]), dtype=tf.float32, name='weight1')
b1 = tf.Variable(tf.random.normal([2]), dtype=tf.float32, name='bias1')
W2 = tf.Variable(tf.random.normal([2, 1], dtype=tf.float32, name='weight2'))
b2 = tf.Variable(tf.random.normal([1]), dtype=tf.float32, name='bias2')

# Nueral network hypothesis: sigmoid(sigmoid(X * W1 + b1) * W2 + b2)
K = tf.nn.sigmoid(tf.linalg.matmul(X, W1) + b1)
hypothesis = tf.nn.sigmoid(tf.linalg.matmul(K, W2) + b2)
prediction = tf.dtypes.cast(hypothesis > 0.5, dtype=tf.float32)

# Define loss using binary cross entropy
loss = tf.reduce_mean(- Y * tf.math.log(hypothesis) - (1 - Y) * tf.math.log(1 - hypothesis))

# Define optimizer and minimize the loss
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1.).minimize(loss)

# Define session and initializer global-variables
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# For loop
for step in range(501):
    _, loss_val = sess.run([optimizer, loss], feed_dict={X: x_data, Y: y_data})

    if step % 50 == 0:
        msg = "Step: {}, Loss: {:.3f}"
        print(msg.format(step, loss_val))

pred_val, label = sess.run([prediction, Y], feed_dict={X: x_data, Y: y_data})
print("Prediction: {}\nLabel: {}".format(pred_val, label))

