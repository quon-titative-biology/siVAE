## Template for saving

import tensorflow as tf

def array_to_tfrecords(X, y, output_file):
    feature = {
    'X': tf.train.Feature(float_list=tf.train.FloatList(value=X.flatten())),
    'y': tf.train.Feature(float_list=tf.train.FloatList(value=y.flatten()))
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    serialized = example.SerializeToString()
    writer = tf.python_io.TFRecordWriter(output_file)
    writer.write(serialized)
    writer.close()


def parse_proto(example_proto):
    features = {
    'X': tf.FixedLenFeature([3], tf.float32)
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    return([parsed_features['X']])


def parse_proto(example_proto):
    features = {
    'X': tf.FixedLenFeature((345,), tf.float32),
    'y': tf.FixedLenFeature((5,), tf.float32),
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    return(parsed_features['X'], parsed_features['y'])


def read_tfrecords(file_names=("file1.tfrecord", "file2.tfrecord", "file3.tfrecord"),
                   buffer_size=10000,
                   batch_size=100):
    dataset = tf.contrib.data.TFRecordDataset(file_names)
    dataset = dataset.map(parse_proto)
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    return(tf.contrib.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes))
