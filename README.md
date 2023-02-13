# High resolution predictions of global snow using recurrent neural networks

[![Deploy](https://github.com/geo-smart/use_case_template/actions/workflows/deploy.yaml/badge.svg)](https://github.com/geo-smart/global_lstm_swe_modeling/actions/workflows/deploy.yaml)
[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://geo-smart.github.io/global_lstm_swe_modeling)
[![GeoSMART Use Case](./book/img/use_case_badge.svg)](https://geo-smart.github.io/usecases)

## Overview
This is a machine learning tutorial that highlights using large-scale climate projections and recurrent neural networks to model snow. In this tutorial we will show you how to train an ML model to evaluate, anywhere in the world, the impact of climate change on snowpack. This tutorial will offer two key novel scientific contributions: (1) using off-the-shelf cloud datasets to produce a model that can generate high resolution snow projections anywhere on the globe in a hindcast setting and (2) evaluating the ability to use the trained model to understand future snow conditions under multiple climate scenarios. Throughout the tutorial we will comment on design decisions which are made when deploying the trained model with the future in mind. This is primarily accomplished by defining modular components for data loading, data transformation, and model loading. As an outcome of this tutorial we hope that you will understand how to apply such methods to your own research.

## Getting started
If you simply want to read the tutorial as a JupyterBook you can click on the Jupyter Book Badge at the top of the readme. If you want to run the tutorial interactively, the easiest way is to use [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/). The tutorial notebooks contained in the `book` directory contains the necessary package installation to run the code.

If you want to run locally, or on custom hardware you will need to install the necessary dependencies yourself. We recommend using conda/mamba to create a python environment to run the tutorial. We have included a base environment that you can start from in the `environment.yml` file, and is based off of the [Pangeo](https://pangeo.io/) environment that Planetary Computer uses.
