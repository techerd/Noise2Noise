# Noise2Noise - Separating internal variability from anthropogenic forcing using neural networks

Overview
--------

Noise2Noise is a set of Python codes developped to separate internal variability from anthropogenic forcing using the Noise2NOise neural networks method (2018 Lehtinen et al., https://doi.org/10.48550/arXiv.1803.04189). This method was applied to climate data by following Bône's work (2024 Bône et al., https://doi.org/10.1029/2023MS003964).

Requirements
-----------
Data are from the ForceSMIP project (Forced Component Estimation Statistical Method Intercomparison Project, https://sites.google.com/ethz.ch/forcesmip/about). The coding environment for Calypso can be found in :/archive/globc/techerd/env_ia as env_ia_sing. 

Documentation 
-------------
Details on the method and results can be found in my report.

Code 
-----------
* **.py**
  
UNet_56.py : code for 1 variable, 1 climate model, annual data

UNet_60.py : code for 1 variable, 5 climate models, annual data

UNet_61.py : code for 1 variable, 5 climate models, monthly data

UNet_68.py : code for 2 variables, 5 climate models, annual data

Generated files can be found in :/archive/globc/techerd/models/ for each U-Net.


* **Paths**

Paths are useful to understand parts of the codes that create datasets.

Normalised data were stored in '/scratch/globc/techer/ia/data_norm/normalisation/variable/model/'
with the name format variable_yearly_model_rmemberi1p1f1.1900-2022_normalisation.npy for annual data and in /scratch/globc/techer/ia/data_norm_monthly/normalisation/variable/model/ as variable_monthly_model_rmemberi1p1f1.190001-202201_normalisation.npy for monthly data.

Member 0 is always the ensemble mean/forced response.
