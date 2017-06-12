import tensorflow as tf


flags = tf.app.flags

# Model parameters

flags.DEFINE_string("flag", "word", "word/char/drmm")


# Training / test parameters
flags.DEFINE_integer("query_len_threshold", 20, "threshold value of query length")
flags.DEFINE_integer("max_bin_size", 30, "bin size of histogram")
flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("num_epochs", 1, "number of epochs")
flags.DEFINE_float("learning_rate", 0.0005, "learning rate")
flags.DEFINE_float("margin", 1.0, "cos margin")

flags.DEFINE_float("validation_steps", 1000, "steps between validations")
flags.DEFINE_float("GPU_rate", 0.9, "steps between validations")

flags.DEFINE_string("training_set", "", "training set path")
flags.DEFINE_string("train_set", "", "train set path")
flags.DEFINE_string("dev_set", "", "dev set path")
flags.DEFINE_string("vocab_path", "", "vocab path")
#flags.DEFINE_string("vectors_path", "../data/vectors_word/vectors_word100_skip_w3_neg50.txt", "vectors path")
flags.DEFINE_string("vectors_path", "", "vectors path")


FLAGS = flags.FLAGS

