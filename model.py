import tensorflow as tf


class Model(object):
    def __init__(self, batch_size, learning_rate, max_query_word, max_bin_size):
        self.num_docs = 2
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_query_word = max_query_word
        self.max_bin_size = max_bin_size
        self._input_layer()
        self.optimizer(histograms=self.histograms, idf=self.idf)
        self.test(histogram=self.histogram, idf=self.idf)
        self.merged_summary_op = tf.summary.merge([self.sm_loss_op])

    def _input_layer(self):
        self.idf = tf.placeholder(dtype=tf.float32, shape=(None, self.max_query_word), name='idf')
        with tf.variable_scope('Train_Inputs'):
            self.histograms = tf.placeholder(dtype=tf.float32,
                                             shape=(None, self.num_docs, self.max_query_word, self.max_bin_size),
                                             name='histogram_maps')

        with tf.variable_scope('Test_Inputs'):
            self.histogram = tf.placeholder(dtype=tf.float32, shape=(None, self.max_query_word, self.max_bin_size),
                                            name='histogram_map')

    def model(self, histogram, idf, is_training=True, reuse=False):
        with tf.variable_scope('model'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            with tf.variable_scope('term_score'):
                # histogram : [batch_size, max_query_word, max_bin_size]
                histogram = tf.reshape(histogram, [-1, self.max_bin_size])
                hidden = tf.layers.dense(inputs=histogram, units=5, activation=tf.nn.relu)
                query_term_score = tf.layers.dense(inputs=hidden, units=1, activation=tf.nn.relu)
                self.query_term_score = tf.reshape(query_term_score, [-1, self.max_query_word])

            with tf.variable_scope('term_gating'):
                w = tf.get_variable("wg", [1], initializer=tf.random_uniform(shape=[1]))
                self.term_gate = tf.exp(w * idf) / tf.reduce_sum(tf.exp(w * idf), axis=-1)

            with tf.variable_scope('match_score'):
                self.match_score = tf.reduce_sum(tf.multiply(self.query_term_score, self.term_gate))

    def optimizer(self, histograms, idf):
        histograms = tf.transpose(histograms, [1, 0, 2, 3], name="histograms_transpose")  # [2, batch_size, max_query_word, max_bin_size]
        self.score_pos = self.model(histogram=histograms[0], idf=idf, is_training=True, reuse=False)  # [batch_size, 1]
        self.score_neg = self.model(histogram=histograms[1], idf=idf, is_training=True, reuse=True)  # [batch_size, 1]

        with tf.name_scope("loss"):
            self.score_pos = tf.squeeze(self.score_pos, -1, name="squeeze_pos")  # [batch_size]
            self.score_neg = tf.squeeze(self.score_neg, -1, name="squeeze_neg")  # [batch_size]
            self.sub = tf.subtract(self.score_pos, self.score_neg, name="pos_sub_neg")

            # self.loss = tf.reduce_mean(tf.maximum(0.0, tf.subtract(1.0, self.sub)))

            self.loss = tf.reduce_mean(tf.log(1.0 + tf.exp(- 1.6 * self.sub)))
            self.sm_loss_op = tf.summary.scalar('Loss', self.loss)

        with tf.name_scope("optimizer"):
            # self.optimize_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            self.optimize_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.90, beta2=0.999,
                                                      epsilon=1e-07).minimize(self.loss)

    def test(self, histogram, idf):
        self.score = self.model(histogram=histogram, idf=idf, is_training=False, reuse=True)
        self.score = tf.squeeze(self.score, axis=-1)
        print("score:", self.score)


