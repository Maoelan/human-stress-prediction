import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from typing import NamedTuple, Dict, Text, Any
from keras_tuner.engine import base_tuner
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs
from stress_trainer import transformed_name, gzip_reader_fn

LABEL_KEY = "label"
FEATURE_KEY = "text"
VOCAB_SIZE = 5000
MAX_TOKEN = 5000
SEQUENCE_LENGTH = 100
OUTPUT_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 16
NUM_EPOCHS = 5
BATCH_SIZE = 64


TunerFnResult = NamedTuple("TunerFnResult", [
    ("tuner", base_tuner.BaseTuner),
    ("fit_kwargs", Dict[Text, Any]),
])


def input_fn(file_pattern, tf_transform_output, num_epochs,
            batch_size=BATCH_SIZE) -> tf.data.Dataset:
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


def model_builder(hp, vectorizer_layer):
    inputs = tf.keras.Input(shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string)
    x = vectorizer_layer(inputs)
    x = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(hp.Choice('unit_1', [64, 128, 256]), activation="relu")(x)
    x = layers.Dense(hp.Choice('unit_2', [32, 64, 128]), activation="relu")(x)
    x = layers.Dense(hp.Choice('unit_3', [16, 32, 64]), activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice('learning_rate', [0.001, 0.0005, 0.0001])
        ),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )
    model.summary()
    return model


def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_set = input_fn(fn_args.train_files[0], tf_transform_output, NUM_EPOCHS)
    eval_set = input_fn(fn_args.eval_files[0], tf_transform_output, NUM_EPOCHS)

    vectorizer_layer = layers.TextVectorization(
        max_tokens=MAX_TOKEN,
        output_mode="int",
        output_sequence_length=OUTPUT_SEQUENCE_LENGTH,
    )
    vectorizer_layer.adapt(train_set.map(lambda f, l: f[transformed_name(FEATURE_KEY)]))

    tuner = kt.RandomSearch(
        hypermodel=lambda hp: model_builder(hp, vectorizer_layer),
        objective="val_binary_accuracy",
        max_trials=20,
        directory=fn_args.working_dir,
        project_name="stress_sentiment_random_search",
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_set,
            "validation_data": eval_set,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
            "epochs": NUM_EPOCHS,
        }
    )