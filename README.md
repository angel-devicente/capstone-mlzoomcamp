
# Table of Contents

1.  [Description of the problem](#org3836813)
    1.  [Dataset](#org6bf849d)
2.  [Preparing environment](#org1db6709)
3.  [Running the project](#org8eb5813)
    1.  [CNN training, etc.](#orge5c3e27)
    2.  [Deployment to AWS Lambda](#org81c244c)
    3.  [Creating a UI](#org896f201)

This repository is the deliverable for the Capstone Project of the Machine
Learning Zoomcamp 2022.


<a id="org3836813"></a>

# Description of the problem

The goal of the project is to use Machine Learning to help predict breast
cancers, based on mammography scans.


<a id="org6bf849d"></a>

## Dataset

The data to be used is the "MIAS Mammography" dataset at Kaggle
(<https://www.kaggle.com/datasets/kmader/mias-mammography>). More information
about the database can be found at <http://peipa.essex.ac.uk/info/mias.html>.

The original data found in the Kaggle dataset had redundant files, and data in a
format that Keras would not be able to use, so I pre-processed them (and store
them in the directory MIAS-data) in the following way:

-   I deleted the files all-mias.tar and all<sub>mias</sub><sub>scans.h5</sub>
-   I converted all .pgm files in directory all-mias to .jpg format. The code used
    to do this conversion was simply:
    
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

There are not a lot of images in this dataset, so probably the prediction will
not be very accurate, but for this project we want to mainly concentrate on the
deployment aspects, so a low prediction rate will not concern us much. On the
contrary, my goal for the final project of the ML-Zoomcamp will be on more
accurate prediction, and not paying so much attention to the delivery aspects.


<a id="org1db6709"></a>

# Preparing environment

In order to fully reproduce the environment (with a given Python version and all
the required requirements):

conda create &#x2013;name capstone<sub>pr</sub> python=3.9.15
conda activate capstone<sub>pr</sub>
pip install pipenv
pipenv install &#x2013;python 3.9.15
pipenv shell

jupyter-notebook &#x2013;no-browser
  open in your favourite browser the URL given
  open the file capstone-project-notebook.ipynb

[remember to test this against the ipykernel of this pipenv and not ml<sub>zoomcamp</sub>,
which I use for development]


<a id="org8eb5813"></a>

# Running the project


<a id="orge5c3e27"></a>

## CNN training, etc.

Could use ideas from:
 <https://www.kaggle.com/code/kmader/pretrained-vgg16-for-mammography-classification/notebook>


<a id="org81c244c"></a>

## Deployment to AWS Lambda


<a id="org896f201"></a>

## Creating a UI

streamlit? <https://streamlit.io/>
<https://joshmantova-eagle-vision-srcproject-eagle-vision-prod-hi8uf8.streamlit.app/>

