import tensorflow as tf

X = [1, 2, 3]
Y = [1, 2, 3]

# Set wrong model weights
W = tf.Variable(5.0, name='weight')
# Linear model
hypothesis = X * W
# Manual gradient
gradient = tf.math.reduce_mean((W * X - Y) * X) * 2
# cost/loss function
cost = tf.math.reduce_mean(tf.math.square(hypothesis - Y))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
# Get gradients
gvs = optimizer.compute_gradients(cost, [W])
# Apply gradietns
apply_gradietns = optimizer.apply_gradients(gvs)

# Launch the graph in a session.
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(100):
    print('step: ', step)
    print(sess.run([gradient, W]))
    print(sess.run(gvs))
