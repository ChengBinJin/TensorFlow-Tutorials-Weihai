import tensorflow as tf

# tf Graph Input
X = [1, 2, 3]
Y = [1, 2, 3]

# Set wrong model weights
W = tf.Variable(-3.0, name='weight')
# Linear model
hypothesis = X * W
# cost/loss function
cost = tf.math.reduce_mean(tf.math.square(hypothesis - Y))
# Minimize: Gradient Descent Magic
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.compat.v1.Session()
# Initializes global variables in the graph.
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(100):
    print(step, sess.run(W))
    sess.run(train)

