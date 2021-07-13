FROM jupyter/scipy-notebook
RUN rmdir work
RUN pip install xgboost
RUN pip3 install torch torchvision torchaudio
COPY ./numerai_datasets/ .
COPY ./numerai.ipynb .