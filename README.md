# Numerai

My attempt at the [numer.ai](https://numer.ai/). My account can be found [here](https://numer.ai/emahony).

## Overview

Using a docker to create a Jupyter notebook where we do some exploratory data analysis.
A basic model exists at the moment.

## Usage

The numerai_datasets.zip must be downloaded first and extracted to the path with the Dockerfile.

## Model Overview

We simply average the 6 feature categories and train a deep neural network on them.
