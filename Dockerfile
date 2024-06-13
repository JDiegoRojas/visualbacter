FROM tensorflow/serving:2.16.1

COPY saved_model/visualbacter /models/visualbacter

ENV MODEL_NAME=visualbacter

EXPOSE 8501