FROM tensorflow/serving:latest

<<<<<<< HEAD
COPY ./serving_model /models
=======
<<<<<<< HEAD
COPY ./serving_model_dir /models
=======
COPY ./serving_model /models
>>>>>>> tuner-experimen
>>>>>>> 22fe7402d7142364c5f433cf633c695da752c4d2
ENV MODEL_NAME=stress-prediction-model