
# Table of Contents

1.  [Description of the problem](#org568218b)
    1.  [Dataset](#org463e9dd)
2.  [Reproducible environment](#org900c92e)
3.  [Running the project](#org23d1f83)

This repository is the deliverable for the Capstone Project of the Machine
Learning Zoomcamp 2022.


<a id="org568218b"></a>

# Description of the problem

The goal of the project is to use Machine Learning to help predict breast
cancers, based on mammography scans.


<a id="org463e9dd"></a>

## Dataset

The data to be used is the "MIAS Mammography" dataset at Kaggle
(<https://www.kaggle.com/datasets/kmader/mias-mammography>). More information
about the database can be found at <http://peipa.essex.ac.uk/info/mias.html>.

There are only 322 images in this dataset, so probably the prediction will not
be very accurate, but for this project I want to mainly concentrate on
reproducibility and deployment aspects, so a low prediction accuracy will not
concern me. On the other hand, my goal for the final project of the ML-Zoomcamp
will be on more accurate prediction, without paying much attention to the
deployment aspects.


<a id="org900c92e"></a>

# Reproducible environment

In order to fully reproduce the environment used when developing this project,
you can use `conda+pipenv`. For the following instructions I assume that you
have `conda` installed and that you are using Linux. If this is
not the case, to install `conda` you can follow the instructions at 
<https://docs.conda.io/en/latest/miniconda.html>. 
*The rest of the instructions shouldn't vary too much if you are using Windows
or macOS, though I haven't been able to test the project in those systems*. With
`conda` properly installed, you should be able to create the environment running
the following commands in the main directory of this repository (i.e. where the
files `Pipfile` and `Pipfile.lock` are located:

    $ conda create --yes --name capstone_pr python=3.9.15
    $ conda activate capstone_pr
    (capstone_pr) $ pip install pipenv
    (capstone_pr) $ pipenv install --python 3.9.15
    (capstone_pr) $ pipenv shell

This will give you the exact version of Python I used while developing the
project (version 3.9.15) and most of the required libraries. 

You can verify that indeed the right library versions have been installed, and
that you are using them by doing, for example:

    (capstone-mlzoomcamp) $ python --version
    Python 3.9.15
    (capstone-mlzoomcamp) $ jupyter --version
    Selected Jupyter core packages...
    IPython          : 8.7.0
    ipykernel        : 6.19.4
    ipywidgets       : 8.0.3
    jupyter_client   : 7.4.8
    jupyter_core     : 5.1.0
    jupyter_server   : 2.0.2
    jupyterlab       : not installed
    nbclient         : 0.7.2
    nbconvert        : 7.2.7
    nbformat         : 5.7.1
    notebook         : 6.5.2
    qtconsole        : 5.4.0
    traitlets        : 5.8.0

But we have to treat tensorflow in a different way, depending on whether we want
to use a GPU for training or not, and we will have to do it manually:

-   tensorflow
    -   if we don't have a GPU, we just do:
        
            (capstone-mlzoomcamp) $ pip install tensorflow
    -   if we have a GPU, we have to do:
        
            (capstone-mlzoomcamp) $ conda install --yes -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
            (capstone-mlzoomcamp) $ conda install --yes -c nvidia cuda-nvcc
            (capstone-mlzoomcamp) $ mkdir -p $CONDA_PREFIX/etc/conda/activate.d
            (capstone-mlzoomcamp) $ echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
            (capstone-mlzoomcamp) $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
            (capstone-mlzoomcamp) $ pip install tensorflow

To open the project notebook, after starting the shell inside the
`(capstone-mlzoomcamp)` environment and after having installed `tensorflow`, you
can start the `jupyter-notebook` in the following way:

    (capstone-mlzoomcamp) $ jupyter-notebook --no-browser

and open the URL provided in the output with your favourite browser. Then just
click on the `capstone-project-notebook.ipynb` file to open the project
notebook. 

If you want to have another shell with this environment just do in the main directory:

    conda activate capstone_pr
    (capstone_pr) $ pipenv shell
    (capstone-mlzoomcamp) $ 

When you are finished with this project, you can simply close the notebook in
your browser, and then stop `jupyter-notebook` (`Ctrl+c`) and then leave the
environment by issuing the following two commands (the first one to exit the
`pipenv` environment, and the second one to deactivate the `conda` environment):

    (capstone-mlzoomcamp) $ exit
    (capstone_pr) $ conda deactivate

If you want to also delete the created virtual environment and the conda
environment, you can run the following commands:

    $ conda activate capstone_pr
    (capstone_pr) $ pipenv --rm
    (capstone_pr) $ conda deactivate
    $ conda env remove -n capstone_pr


<a id="org23d1f83"></a>

# Running the project

The notebook `capstone-project-notebook.ipynb` has a detailed description of the
data preparation and exploration, as well as the convolutional neural network
prepared to predict abnormalities in mammographies. 

It also describes how to locally train and test the model using TensorFlow
Keras, as well as how to deploy the model for the AWS Lambda service (testing it
locally first with the help of TensorFlow-Lite and Docker).

