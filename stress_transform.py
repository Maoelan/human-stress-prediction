
import tensorflow as tf

LABEL_KEY = "label"
FEATURE_KEY = "text"

def transformed_name(key) :
    return key + "_xf"

def preprocessing_fn(inputs) :
    outputs = {}
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(inputs[FEATURE_KEY])
    return outputs
