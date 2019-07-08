import numpy as np

xy = np.loadtxt('./data/examdata.txt', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# Make sure the shape and data are OK
print('x_data shape: {}'.format(x_data.shape))
print('x_data: {}'.format(x_data))
print('len(x_data): {}'.format(len(x_data)))
print('y_data.shape: {}'.format(y_data.shape))
print('y_data: {}'.format(y_data))

