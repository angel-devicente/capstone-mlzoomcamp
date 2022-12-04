
# Table of Contents

1.  [Description of the problem](#orgcfeb0b8)
    1.  [Dataset](#org21684d0)
2.  [Preparing environment](#org65da851)
3.  [Running the project](#org6f69822)
    1.  [CNN training, etc.](#org6aa840b)
    2.  [Deployment to AWS Lambda](#orgef1b17f)
    3.  [Creating a UI](#orgfbd604d)

This repository is the deliverable for the Capstone Project of the Machine
Learning Zoomcamp 2022.


<a id="orgcfeb0b8"></a>

# Description of the problem

The goal of the project is to use Machine Learning to help predict breast
cancers, based on mammography scans.


<a id="org21684d0"></a>

## Dataset

The data to be used is the "MIAS Mammography" dataset at Kaggle
(<https://www.kaggle.com/datasets/kmader/mias-mammography>). These data can be
found in the directory MIAS-data, though I have removed some redundant files
from the available Kaggle data to save space. More information about the
database can be found at <http://peipa.essex.ac.uk/info/mias.html>.

[The data in .h5 is redundant as well, as all the data we need is Info.txt and
the all-mias directory with all the scans.]

There are not a lot of images in this dataset, so probably the prediction will
not be very accurate, but for this project we want to mainly concentrate on the
deployment aspects, so a low prediction rate will not concern us much. On the
contrary, my goal for the final project of the ML-Zoomcamp will be on more
accurate prediction, and not paying so much attention to the delivery aspects.


<a id="org65da851"></a>

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


<a id="org6f69822"></a>

# Running the project


<a id="org6aa840b"></a>

## CNN training, etc.

Could use ideas from:
 <https://www.kaggle.com/code/kmader/pretrained-vgg16-for-mammography-classification/notebook>


<a id="orgef1b17f"></a>

## Deployment to AWS Lambda


<a id="orgfbd604d"></a>

## Creating a UI

streamlit? <https://streamlit.io/>
<https://joshmantova-eagle-vision-srcproject-eagle-vision-prod-hi8uf8.streamlit.app/>

