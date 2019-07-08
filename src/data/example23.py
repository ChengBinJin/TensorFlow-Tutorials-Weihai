import tensorflow as tf

x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_and = [[0], [0], [0], [1]]
y_or = [[0], [1], [1], [1]]
y_xor = [[0], [1], [1], [0]]

X = tf.compat.v1.placeholder(tf.float32, shape=[None, 2], name='X')
Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='Y')

W = tf.Variable(tf.random.normal([2, 1]), dtype=tf.float32, name='weight')
b = tf.Variable(tf.random.normal([1]), dtype=tf.float32, name='bias')

# hypotehsis
hypothesis = tf.linalg.matmul(X, W) + b
logits = tf.nn.sigmoid(hypothesis)

loss = tf.math.reduce_mean(- Y * tf.math.log(logits) - (1 - Y) * tf.math.log(1 - logits))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(100):
    _, loss_val = sess.run([optimizer, loss], feed_dict={X: x_data, Y: y_and})

# 16, 15:00-18:00
# 18, 15:00-18:00