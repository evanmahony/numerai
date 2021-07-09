FROM jupyter/scipy-notebook
RUN rmdir work
RUN pip3 install torch torchvision torchaudio numerapi xgboost
COPY ./src/ .
RUN python init.py
RUN mv ./numerai_datasets/numerai_training_data.csv ./numerai_datasets/numerai_tournament_data.csv .