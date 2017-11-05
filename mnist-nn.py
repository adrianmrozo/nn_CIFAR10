import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes =10  # MNIST dataset classes 0-9
batch_size = 200 # the number of training examples in one forward/backward pass

# rows = batch size, columns = flattened image

x = tf.placeholder('float' ,[None,784])
y = tf.placeholder('float')

def neural_network_model(data):

	# initializing weights and biases for the hidden layers and the output layer
	hidden_l1 = {'weights': tf.Variable(tf.random_normal([784,n_nodes_hl1])), 
				 'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_l2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])), 
				 'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_l3 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])), 
				 'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_l = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])), 
				 'biases':tf.Variable(tf.random_normal([n_classes]))}

	# computing the ouput of each layer and applying a non-linearity at the end of each layer computation
	l1 = tf.add(tf.matmul(data, hidden_l1['weights']), hidden_l1['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_l2['weights']), hidden_l2['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_l3['weights']), hidden_l3['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.add(tf.matmul(l3, output_l['weights']), output_l['biases'])

	return output


def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

	# learning rate = 0.001
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	# one epoch = one cycle of feed-forward and backprop
	n_epochs = 20 

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(n_epochs):
			epoch_loss = 0
			for _ in range (int(mnist.train.num_examples/batch_size)):
				x_epoch,y_epoch = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict = {x:x_epoch, y:y_epoch}) 
				epoch_loss +=c 
			print ('Epoch', epoch, 'completed out of', n_epochs, 'loss:', epoch_loss)

		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy  = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)
