import numpy as np
import tensorflow as tf

# Predicting animal type based on various features
xy = np.loadtxt('./data/animalzoo.txt', delimiter=',', dtype=np.float32)
x_data = xy[:, :-1]
y_data = xy[:, [-1]]

print('x_data shape: {}'.format(x_data.shape))
print('y_data shape: {}'.format(y_data.shape))

nb_classes = int(np.amax(y_data) + 1)  # 0 ~ 6
X = tf.compat.v1.placeholder(tf.float32, shape=[None, x_data.shape[1]], name='X')
Y = tf.compat.v1.placeholder(tf.int32, shape=[None, 1], name='Y')  # 0 ~ 6, shape=(?, 1)
Y_one_hot = tf.one_hot(Y, depth=nb_classes)  # one hot shape=(?, 1, 7)
Y_one_hot = tf.reshape(Y_one_hot, shape=[-1, nb_classes])  # shape=(?, 7)

W = tf.Variable(tf.random.normal([x_data.shape[1], nb_classes]), name='weight')
b = tf.Variable(tf.random.normal([nb_classes]), name='bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
logits = tf.linalg.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

# Softmax cross entropy loss/cost
cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y_one_hot)
cost = tf.math.reduce_mean(cost_i)
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.math.argmax(hypothesis, axis=1)
correct_prediction = tf.math.equal(prediction, tf.math.argmax(Y_one_hot, axis=1))
accuracy = tf.math.reduce_mean(tf.dtypes.cast(correct_prediction, dtype=tf.float32)) * 100.

# Launch graph
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(2001):
    _, loss_val, acc_val = sess.run([train, cost, accuracy], feed_dict={X: x_data, Y: y_data})
    if step % 100 == 0:
        msg = "Step: {}, loss: {:.3f}, Acc: {:.2f}"
        print(msg.format(step, loss_val, acc_val))

# Let's see if we can predict
pred = sess.run(prediction, feed_dict={X: x_data})
# y_data: (N, 1) = flatten => (N, ) matches pred.shape
for p, y in zip(pred, y_data.flatten().astype(np.uint8)):
    print("[{}] Prediction: {}, GT: {}".format(p == y, p, y))

