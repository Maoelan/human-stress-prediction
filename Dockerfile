FROM tensorflow/serving:latest

<<<<<<< HEAD
COPY ./serving_model_dir /models
=======
COPY ./serving_model /models
>>>>>>> tuner-experimen
ENV MODEL_NAME=stress-prediction-model