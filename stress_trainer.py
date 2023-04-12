import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers
import os
import tensorflow_hub as hub
from tfx.components.trainer.fn_args_utils import FnArgs

LABEL_KEY = "label"
FEATURE_KEY = "text"

def transformed_name(key) :
    return key + "_xf"

def gzip_reader_fn(filenames) :
    return tf.data.TFRecordDataset(filenames, compression_type = 'GZIP')

def input_fn(file_pattern,
            tf_transform_output,
            num_epochs,
            batch_size=64)->tf.data.Dataset:

            transform_feature_spec = (
                tf_transform_output.transformed_feature_spec().copy()
            )

            dataset = tf.data.experimental.make_batched_features_dataset(
                file_pattern = file_pattern,
                batch_size = batch_size,
                features = transform_feature_spec,
                reader = gzip_reader_fn,
                num_epochs = num_epochs,
                label_key = transformed_name(LABEL_KEY)
            )
            return dataset

VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 100

vectorize_layer = layers.TextVectorization(
    standardize = "lower_and_strip_punctuation",
    max_tokens = VOCAB_SIZE,
    output_mode = 'int',
    output_sequence_length = SEQUENCE_LENGTH
)

embedding_dim = 16

def model_builder() :
    inputs = tf.keras.Input(shape=(1,), name = transformed_name(FEATURE_KEY), dtype = tf.string)
    reshaped_narrative = tf.reshape(inputs, [-1])
    x = vectorize_layer(reshaped_narrative)
    x = layers.Embedding(VOCAB_SIZE, embedding_dim, name = "embedding")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation = "relu")(x)
    x = layers.Dense(32, activation = "relu")(x)
    outputs = layers.Dense(1, activation = "sigmoid")(x)

    model = tf.keras.Model(inputs = inputs, outputs = outputs)

    model.compile(
        loss = 'binary_crossentropy',
        optimizer = tf.keras.optimizers.Adam(0.01),
        metrics = [tf.keras.metrics.BinaryAccuracy()]
    )

    model.summary()
    return model

def _get_serve_tf_examples_fn(model, tf_transform_output) :

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples) :
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)
    return serve_tf_examples_fn

def run_fn(fn_args : FnArgs) -> None :
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir = log_dir, update_freq = 'batch'
    )

    es = tf.keras.callbacks.EarlyStopping(monitor = 'val_binary_accuracy', mode = 'max', verbose = 1, patience = 10)
    mc = tf.keras.callbacks.ModelCheckpoint(fn_args.serving_model_dir, monitor = 'val_binary_accuracy', mode = 'max', verbose=1, save_best_only=True)

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_set = input_fn(fn_args.train_files, tf_transform_output, 10)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, 10)
    vectorize_layer.adapt(
        [j[0].numpy()[0] for j in [
            i[0][transformed_name(FEATURE_KEY)]
                for i in list(train_set)]
        ]
    )

    model = model_builder()

    model.fit(x = train_set,
            validation_data = val_set,
            callbacks = [tensorboard_callback, es, mc],
            steps_per_epoch = 1000,
            validation_steps = 1000,
            epochs = 10
    )

    signatures = {
        'serving_default' :
        _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
                                tf.TensorSpec(
                                    shape = [None],
                                    dtype = tf.string,
                                    name = 'examples'
                                )
        )
    }
    
    model.save(fn_args.serving_model_dir, save_format = 'tf', signatures = signatures)
