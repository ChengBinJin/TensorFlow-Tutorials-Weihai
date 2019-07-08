import tensorflow as tf

x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4],
          [4, 1, 5, 5], [1, 7, 5, 5], [1, 2, 5, 6],
          [1, 6, 6, 6], [1, 7, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1],
          [0, 1, 0], [0, 1, 0], [0, 1, 0],
          [1, 0, 0], [1, 0, 0]]

X = tf.compat.v1.placeholder(tf.float32, shape=[None, 4], name='X')
Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3], name='Y')
W = tf.Variable(tf.random.normal([4, 3]), dtype=tf.float32, name='weight')
b = tf.Variable(tf.random.normal([3]), dtype=tf.float32, name='bias')
hypothesis = tf.nn.softmax(tf.linalg.matmul(X, W) + b)
cost = tf.math.reduce_mean(tf.math.reduce_sum(- Y * tf.math.log(hypothesis), axis=1))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1.5).minimize(cost)

# Correct prediction Test model
prediction = tf.math.argmax(hypothesis, axis=1)
is_correct = tf.math.equal(prediction, tf.math.argmax(Y, axis=1))
accuracy = tf.math.reduce_mean(tf.dtypes.cast(is_correct, dtype=tf.float32))
# Launch graph
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(201):
    cost_val, W_val, _ = sess.run([cost, W, optimizer], feed_dict={X: x_data, Y: y_data})
    print(step, cost_val, W_val)

# Predict
print("Prediction: {}".format(sess.run(prediction, feed_dict={X: x_data})))
# Calculate teh accuracy
print("Accuracy: {}".format(sess.run(accuracy, feed_dict={X: x_data, Y: y_data})))

