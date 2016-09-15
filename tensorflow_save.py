import tensorflow as tf

class TFObject:

    def __init__(self):
        # Some train parameters
        self.learning_rate = 0.001

        # Some network parameters
        self.n_input = 500  # input features
        self.n_classes = 2  # output classes
        self.x = tf.placeholder("float", [None, self.n_input], name="nX")
        self.y = tf.placeholder("float", [None, self.n_classes], name="nY")

        self.weights = {
            'h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
            'out': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_classes]))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }
        # Multilayer perceptron:
        # one hidden layer with RELU activation
        self.pTron = tf.add(tf.matmul(tf.nn.relu(tf.add(tf.matmul(self.x, self.weights['h1']), self.biases['b1'])), self.weights['out']), self.biases['out'], name="nPTron")
        self.init = tf.initialize_variables(tf.all_variables(), name="nInit")

        self.saver = tf.train.Saver(name="nSaver")

        # Save variables from disk.
        with tf.Session() as sess:
            self.saver.save(sess, "/tmp/model.ckpt")

    def save(self, filename):
        for variable in tf.trainable_variables():
            tensor = tf.constant(variable.eval())
            tf.assign(variable, tensor, name="nWeights")

        # This does not work in tensorflow with python3 now,
        # but we defenetely need to save graph as binary!
        tf.train.write_graph(self.sess.graph_def, 'graph/', 'graph.pb', as_text=False)