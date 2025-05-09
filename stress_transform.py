<<<<<<< HEAD
import tensorflow as tf
import tensorflow_transform as tft
=======
<<<<<<< HEAD

import tensorflow as tf
=======
import tensorflow as tf
import tensorflow_transform as tft
>>>>>>> tuner-experimen
>>>>>>> 22fe7402d7142364c5f433cf633c695da752c4d2

LABEL_KEY = "label"
FEATURE_KEY = "text"

<<<<<<< HEAD
=======
<<<<<<< HEAD
def transformed_name(key) :
    return key + "_xf"
>>>>>>> 22fe7402d7142364c5f433cf633c695da752c4d2

def transformed_name(key):
    return f"{key}_xf"


def preprocessing_fn(inputs):
    outputs = {}
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(inputs[FEATURE_KEY])
<<<<<<< HEAD
    return outputs
=======
    return outputs
=======

def transformed_name(key):
    return f"{key}_xf"


def preprocessing_fn(inputs):
    outputs = {}
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(inputs[FEATURE_KEY])
    return outputs
>>>>>>> tuner-experimen
>>>>>>> 22fe7402d7142364c5f433cf633c695da752c4d2
