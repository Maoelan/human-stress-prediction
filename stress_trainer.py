import os
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tfx.components.trainer.fn_args_utils import FnArgs
import json

LABEL_KEY = "label"
FEATURE_KEY = "text"
VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 16
NUM_EPOCHS = 10
BATCH_SIZE = 64


def transformed_name(key):
    return f"{key}_xf"


def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=BATCH_SIZE) -> tf.data.Dataset:
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY),
    )
    return dataset


vectorize_layer = layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH,
)


def model_builder(hp):
    if isinstance(hp, dict) and 'values' in hp:
        hp = hp['values']

    inputs = tf.keras.Input(shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string)
    reshaped_narrative = tf.reshape(inputs, [-1])
    x = vectorize_layer(reshaped_narrative)
    x = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, name="embedding")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(hp['unit_1'], activation="relu")(x)
    x = layers.Dense(hp['unit_2'], activation="relu")(x)
    x = layers.Dense(hp['unit_3'], activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp['learning_rate']
        ),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )

    model.summary()
    return model


def _get_serve_tf_examples_fn(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_tf_examples_fn


def run_fn(fn_args: FnArgs) -> None:
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq="batch"
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_binary_accuracy", mode="max", verbose=1, patience=10
    )
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        fn_args.serving_model_dir,
        monitor="val_binary_accuracy",
        mode="max",
        verbose=1,
        save_best_only=True,
    )
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, verbose=1)

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_set = input_fn(fn_args.train_files, tf_transform_output, 10)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, 10)

    vectorize_layer.adapt(
        [
            j[0].numpy()[0]
            for j in [
                i[0][transformed_name(FEATURE_KEY)] for i in list(train_set)
            ]
        ]
    )

    best_hyperparameters = fn_args.hyperparameters
    if isinstance(best_hyperparameters, str):
        best_hyperparameters = json.loads(best_hyperparameters)

    model = model_builder(best_hyperparameters)

    model.fit(
        x=train_set,
        validation_data=val_set,
        callbacks=[tensorboard_callback, early_stopping, model_checkpoint, reduce_lr],
        steps_per_epoch=1000,
        validation_steps=1000,
        epochs=NUM_EPOCHS,
    )

    signatures = {
        "serving_default": _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        )
    }

    model.save(fn_args.serving_model_dir, save_format="tf", signatures=signatures)