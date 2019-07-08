import tensorflow as tf

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
# Placeholders for a tensor that will be always fed.
X = tf.compat.v1.placeholder(tf.float32, shape=[None, 2], name='X')
Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='Y')

W = tf.Variable(tf.random.normal([2, 1], name='weight'))
b = tf.Variable(tf.random.normal([1]), name='bias')
# Hypothesis using sigmoid: tf.div(1., 1. + tf.math.exp(-tf.linalg.matmul(X, W)))
hypothesis = tf.math.sigmoid(tf.linalg.matmul(X, W) + b)
# cost/loss function
cost = -tf.math.reduce_mean(Y * tf.math.log(hypothesis) + (1 - Y) * tf.math.log(1 - hypothesis))
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.dtypes.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.math.reduce_mean(tf.dtypes.cast(tf.math.equal(predicted, Y), dtype=tf.float32))

# Launch graph
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(10001):
    cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
    if step % 200 == 0:
        print(step, cost_val)

# Accuracy report
hyp_val, pre_val, acc_val = sess.run([hypothesis, predicted, accuracy],
                                     feed_dict={X: x_data, Y: y_data})
print("\nHypothesis: ", hyp_val, "\nPredict (Y): ", pre_val, "\nAccuracy: ", acc_val)
