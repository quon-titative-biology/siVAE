import h5py
import tensorflow as tf

num_feature = 5

def parse_proto(example_proto):
    features = {
        # 'X': tf.FixedLenFeature((num_feature,), tf.float32, allow_missing = True)
        # 'X': tf.FixedLenFeature((num_feature,), tf.float32)
        'X': tf.compat.v1.FixedLenFeature([num_feature], tf.float32)
        # 'X': tf.FixedLenFeature([], tf.float32)
    }
    # parsed_features = tf.compat.v1.parse_single_example(example_proto, features)
    parsed_features = tf.io.parse_single_example(serialized = example_proto,
                                                        features   = features)
    return([parsed_features['X']])

class parser():
    """ """
    def __init__(self, features = {}):
        self.features = features

    def parse(self, example):
        parsed_features = tf.io.parse_single_example(serialized = example,
                                                     features   = self.features)
        return([value for key,value in parsed_features.items()])
