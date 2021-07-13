FROM python
RUN pip install tensorflow
COPY . . 
RUN mkdir logs
RUN unzip -d logs log_set-2021-03-11.zip
CMD ["tensorboard", "--logdir", "logs", "--host", "0.0.0.0"]