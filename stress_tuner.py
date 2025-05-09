import os
import tensorflow as tf
import tensorflow_transform as tft
import kerastuner as kt
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.v1.components import TunerFnResult
from trainer import transformed_name, LABEL_KEY, FEATURE_KEY, input_fn

VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 100

def model_builder(hp):
    vectorize_layer = tf.keras.layers.TextVectorization(
        standardize="lower_and_strip_punctuation",
        max_tokens=VOCAB_SIZE,
        output_mode='int',
        output_sequence_length=SEQUENCE_LENGTH
    )

    embedding_dim = hp.Choice('embedding_dim', [8, 16, 32])
    dense_units_1 = hp.Choice('units_1', [32, 64, 128])
    dense_units_2 = hp.Choice('units_2', [16, 32, 64])
    learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 5e-4])

    inputs = tf.keras.Input(shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string)
    reshaped_inputs = tf.reshape(inputs, [-1])
    x = vectorize_layer(reshaped_inputs)
    x = tf.keras.layers.Embedding(VOCAB_SIZE, embedding_dim)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(dense_units_1, activation="relu")(x)
    x = tf.keras.layers.Dense(dense_units_2, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def tuner_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = input_fn(fn_args.train_files, tf_transform_output, num_epochs=1, batch_size=32)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=1, batch_size=32)

    tuner = kt.RandomSearch(
        model_builder,
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory=fn_args.working_dir,
        project_name='text_binary_tuner'
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_dataset,
            "validation_data": eval_dataset,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
            "epochs": 10
        }
    )