FROM jupyter/scipy-notebook
RUN rmdir work
RUN mkdir data runs
COPY --chown=jovyan ./src/ .
RUN pip3 install torch torchvision torchaudio numerapi xgboost datatable tensorboard
RUN python init.py
RUN mv ./numerai_datasets/numerai_training_data.csv ./numerai_datasets/numerai_tournament_data.csv ./data/