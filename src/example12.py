import tensorflow as tf

x_data = [[73., 80., 75.], [93., 88., 93.],
          [89., 91., 90.], [96., 98., 100.], [73., 66., 70.]]
y_data = [[152.], [185.], [180.], [196.], [142.]]

# Placeholders for a tensor that will be always fed.
X = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random.normal([3, 1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

# Hypothesis
hypothesis = tf.linalg.matmul(X, W) + b
# Simplified cost/loss function
cost = tf.math.reduce_mean(tf.math.square(hypothesis - Y))
# Minimize
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.compat.v1.Session()
# Initializes global variables in the graph.
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                   feed_dict={X: x_data, Y: y_data})

    if step % 20 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

