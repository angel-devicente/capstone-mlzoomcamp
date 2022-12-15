
# Table of Contents

1.  [Description of the problem](#org74497da)
    1.  [Dataset](#org1b6afc2)
2.  [Reproducible environment](#org6366ba5)
3.  [Running the project](#org08a46ba)
    1.  [CNN training, etc.](#org0acf489)
    2.  [Deployment to AWS Lambda](#org6fe642e)
    3.  [Creating a UI](#org73ee75a)

This repository is the deliverable for the Capstone Project of the Machine
Learning Zoomcamp 2022.


<a id="org74497da"></a>

# Description of the problem

The goal of the project is to use Machine Learning to help predict breast
cancers, based on mammography scans.


<a id="org1b6afc2"></a>

## Dataset

The data to be used is the "MIAS Mammography" dataset at Kaggle
(<https://www.kaggle.com/datasets/kmader/mias-mammography>). More information
about the database can be found at <http://peipa.essex.ac.uk/info/mias.html>.

The original data found in the Kaggle dataset had redundant files, and data in a
format that Keras would not be able to use, so I pre-processed them (and store
them in the directory `MIAS-data`) in the following way:

-   I deleted the files `all-mias.tar` and `all_mias_scans.h5`
-   I converted all `.pgm` files in directory `all-mias` to `.jpg` format. The
    code used to do this conversion was simply:
    
        import os
        from PIL import Image
        
        directory = 'MIAS-data/all-mias'
        
        for file in os.listdir(directory):
            f = os.path.join(directory, file)
            if file.endswith('.pgm') and os.path.isfile(f): 
                nfile = file.replace('.pgm','.jpg')
                nf = os.path.join(directory, nfile)    
                img = Image.open(f)
                img.save(nf)

There are only 322 images in this dataset, so probably the prediction will not
be very accurate, but for this project we want to mainly concentrate on the
reproducibility and deployment aspects, so a low prediction rate will not
concern us much. On the contrary, my goal for the final project of the
ML-Zoomcamp will be on more accurate prediction, without paying so much
attention to the delivery aspects.


<a id="org6366ba5"></a>

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

    conda create --name capstone_pr python=3.9.15
    conda activate capstone_pr
    pip install pipenv
    pipenv install --python 3.9.15
    pipenv shell

This will give you the exact version of Python I used while developing the
project (version 3.9.15) and all the required libraries. 

You can verify that indeed the right library versions have been installed, and
that you are using them by doing, for example:

    (capstone-mlzoomcamp) $ python --version
    Python 3.9.15
    (capstone-mlzoomcamp) $ jupyter --version
    Selected Jupyter core packages...
    IPython          : 8.7.0
    ipykernel        : 6.17.1
    ipywidgets       : 8.0.2
    jupyter_client   : 7.4.7
    jupyter_core     : 5.1.0
    jupyter_server   : 1.23.3
    jupyterlab       : not installed
    nbclient         : 0.7.2
    nbconvert        : 7.2.5
    nbformat         : 5.7.0
    notebook         : 6.5.2
    qtconsole        : 5.4.0
    traitlets        : 5.6.0

When you are finished with this project, you can leave the environment by
issuing the following two commands (the first one to exit the `pipenv`
environment, and the second one to deactivate the `conda` environment):

    (capstone-mlzoomcamp) $ exit
    (capstone_pr) $ conda deactivate

pipenv shell
jupyter-notebook &#x2013;no-browser
  open in your favourite browser the URL given
  open the file capstone-project-notebook.ipynb

[remember to test this against the ipykernel of this pipenv and not ml<sub>zoomcamp</sub>,
which I use for development]


<a id="org08a46ba"></a>

# Running the project


<a id="org0acf489"></a>

## CNN training, etc.

Could use ideas from:
 <https://www.kaggle.com/code/kmader/pretrained-vgg16-for-mammography-classification/notebook>


<a id="org6fe642e"></a>

## Deployment to AWS Lambda


<a id="org73ee75a"></a>

## Creating a UI

streamlit? <https://streamlit.io/>
<https://joshmantova-eagle-vision-srcproject-eagle-vision-prod-hi8uf8.streamlit.app/>

