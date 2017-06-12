#!/usr/bin/env python
import os
import time
import numpy as np
import tensorflow as tf
from util import FLAGS

from model import Model
from load_data import  LoadTrainData, LoadTestData
from tester import test

training_set = LoadTrainData(data_path=FLAGS.training_set,
                             batch_size=FLAGS.batch_size)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = FLAGS.GPU_rate

with tf.Session(config=config) as sess:
    timestamp = str(int(time.time()))
    print("timestamp: ", timestamp)
    model_name = "{}_lr{}_bz{}_mg{}_{}".format(FLAGS.flag,
                                            FLAGS.learning_rate,
                                            FLAGS.batch_size,
                                            FLAGS.margin,
                                            timestamp)

    model = Model(learning_rate=FLAGS.learning_rate,
                  batch_size=FLAGS.batch_size,
                  max_query_word=FLAGS.query_len_threshold,
                  max_bin_size=FLAGS.max_bin_size)

    log_dir = "../logs/" + model_name
    model_path = os.path.join(log_dir, "model.ckpt")
    train_writer = tf.summary.FileWriter(log_dir + "/train", sess.graph)

    init = tf.global_variables_initializer()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    step = 0
    num_epochs = FLAGS.num_epochs
    for epoch in range(num_epochs):
        print("epoch: ", epoch)
        #for i in range(10):
        #    features_local, queries, docs = sess.run([features_local_batch, query_batch, docs_batch])
        for batch_data in training_set.next_batch():
            histograms, idfs = batch_data

            feed_dict = {model.histograms: histograms,
                         model.idf: idfs}

            _, loss, score_pos, score_neg, subs, summary =\
                sess.run([model.optimize_op, model.loss, model.score_pos,
                          model.score_neg, model.sub, model.merged_summary_op],
                         feed_dict)

            train_writer.add_summary(summary, step)

            if step % FLAGS.validation_steps == 0:
                print(step, " - loss:", loss)
                train_set = LoadTestData(FLAGS.dev_set, batch_size=FLAGS.batch_size)
                print("On training set:\n")
                dcg_3, dcg_5, dcg_full = test(sess, model, train_set, filename="train_result.csv")

                dev_set = LoadTestData(FLAGS.dev_set, batch_size=FLAGS.batch_size)
                print("On validation set:\n")

                dcg_3, dcg_5, dcg_full = test(sess, model, dev_set, filename="dev_result.csv")

            step += 1
        #saver = tf.train.Saver(tf.global_variables())
        #saver_path = saver.save(sess, os.path.join(log_dir, "model.ckpt"), step)

    train_writer.close()

    coord.request_stop()
    coord.join(threads)
