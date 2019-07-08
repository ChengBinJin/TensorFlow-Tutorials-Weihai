import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from mnist import MNIST

data = MNIST(data_dir='./data/MNIST_data')
print("Size of:")
print("- Training-set:\t\t{}".format(data.num_train))
print("- Validation-set:\t{}".format(data.num_val))
print("- Test-set:\t\t{}".format(data.num_test))

# The images are stored in one-dimensional arrays of this length.
img_size_flat = data.img_size_flat
print('- img_size_flat:\t{}'.format(img_size_flat))

# tuple with height and width of images used to reshape arrays.
img_shape = data.img_shape
print('- img_shape:\t\t{}'.format(img_shape))

# Number of classes, one class for each of 10 digits.
num_classes = data.num_classes
print('- num_classes:\t\t{}'.format(num_classes))

print(data.y_test[0:5, :])
print(data.y_test_cls[0:5])

# MNIST data image of shape, img_size_flat: 28 * 28 = 784
X = tf.compat.v1.placeholder(tf.float32, shape=[None, img_size_flat], name='X')
# 0 - 9 digits recognition, num_classes = 10 classes
Y = tf.compat.v1.placeholder(tf.float32, shape=[None, num_classes], name='Y')

W = tf.Variable(tf.random.normal([img_size_flat, num_classes]), dtype=tf.float32, name='weight')
b = tf.Variable(tf.random.normal([num_classes]), dtype=tf.float32, name='bias')

# Hypothesis (using softmax)
hypothesis = tf.linalg.matmul(X, W) + b
pred_cls = tf.math.argmax(hypothesis, axis=1)

cost = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Test model
is_correct = tf.math.equal(tf.math.argmax(hypothesis, axis=1), tf.math.argmax(Y, axis=1))
# Calculate accuracy
accuracy = tf.math.reduce_mean(tf.dtypes.cast(is_correct, dtype=tf.float32))

# Parameters
training_epochs = 15
batch_size = 100

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epoch in range(training_epochs):
    avg_cost = 0.
    total_iterations = int(data.num_train / batch_size)

    for iteration in range(total_iterations):
        x_batch, y_batch, y_batch_cls = data.random_batch(batch_size=batch_size)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: x_batch, Y: y_batch})
        avg_cost += cost_val / total_iterations

    print('Epoch: {:2d}, cost={:.3f}'.format(epoch + 1, avg_cost))
print('Learning finished')

# Test the model using test sets
acc_val = sess.run(accuracy, feed_dict={X: data.x_test, Y:data.y_test})
print("Test accuracy: {:.2%}".format(acc_val))

# Get some images and predict them
num_test_imgs = 5
indexes = np.random.randint(low=0, high=data.num_test, size=num_test_imgs)
labels = data.y_test_cls[indexes]
predictions = sess.run(pred_cls, feed_dict={X: data.x_test[indexes]})

# Show image
for i, idx in enumerate(indexes):
    plt.imshow(data.x_test[idx].reshape(data.img_shape), cmap='Greys', interpolation='nearest')
    plt.xlabel("Label: {}\nPrediction: {}".format(labels[i], predictions[i]))
    plt.xticks([])  # turn off x ticks
    plt.yticks([])  # turn off y ticks
    plt.show()  # show image

