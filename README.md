## About

The repository contains the software used to obtain the DeepHistReg results (ANHIR challenge).
The purpose of the repository is to make the results fully reproducible. Please contact us in case of any technical problems.
For the iterative MIND-Demons (the best median of median rTRE among all participants but significantly higher registration time) please visit: [ANHIR-AGH](https://github.com/lNefarin/ANHIR_MW)

## How to use

There are two ways to use the code:
* To reproduce the results 
* To improve the registration procedure and perform your own experiments 

## Access to the submission file

If you do not want or do not have time to use the code and the purpose is just to obtain the results just contact us.
We will provide the submission file used to create the ANHIR submission with the transformed landmarks and the reported registration time.

## How to reproduce the results

To reproduce the results without performing the training from scratch, perform the following steps:

* [Prepare Dataset](https://github.com/lNefarin/DeepHistReg/blob/master/prepare_datasets.py)
* Use the file to convert the original ANHIR dataset into the format used by the DeepHistReg framework.
* Contact authors for the access to the pretrained models and use: [Main File](https://github.com/lNefarin/DeepHistReg/blob/master/main.py).
* First point the model variables to appropriate paths (documented in the file) and then run the file.
* Create the submission from the registration output using [Submission File](https://github.com/lNefarin/DeepHistReg/blob/master/prepare_submission.py).
* Add your own machine benchmark file to the submission folder and zip the folder (see challenge website [ANHIR Webiste](https://anhir.grand-challenge.org/Data/)).
* The submission is ready.

## How to perform own traning/validation

To improve the results and perform your own training, the following steps are necessary:
* [Prepare Dataset](https://github.com/lNefarin/DeepHistReg/blob/master/prepare_datasets.py)
* Use the file to convert the original ANHIR dataset into the format used by the DeepHistReg framework.
* Use [Main File](https://github.com/lNefarin/DeepHistReg/blob/master/main.py) with only the initial alignment to create training set for the affine registration. Then use [Prepare Dataset](https://github.com/lNefarin/DeepHistReg/blob/master/prepare_datasets.py) to prepare the dataset for the affine training.
* Use [Affine Reg](https://github.com/lNefarin/DeepHistReg/blob/master/affine_registration.py) to train the affine registration network.
* Use [Main File](https://github.com/lNefarin/DeepHistReg/blob/master/main.py) with the initial alignment and the affine registration to create the training set for the nonrigid registration. Then use [Prepare Dataset](https://github.com/lNefarin/DeepHistReg/blob/master/prepare_datasets.py) to prepare the dataset for the nonrigid training.
* Use [Nonrigid Reg](https://github.com/lNefarin/DeepHistReg/blob/master/deformable_registration.py) to train the nonrigid registration network.
* Use [Main File](https://github.com/lNefarin/DeepHistReg/blob/master/main.py) with the whole framework turned on to create the final registration results.
* Create the submission from the registration output using [Submission File](https://github.com/lNefarin/DeepHistReg/blob/master/prepare_submission.py).
* Add your own machine benchmark file to the submission folder and zip the folder (see challenge website [ANHIR Webiste](https://anhir.grand-challenge.org/Data/)).
* The submission is ready.

## Dataset

For the dataset access and full description please visit [ANHIR Webiste](https://anhir.grand-challenge.org/Data/).
If you found the dataset useful please cite the appropriate publications.

## Dependencies

* PyTorch
* NumPy
* SciPy
* Matplotlib
* SimpleITK
* Pandas
* PyTorch-Summary

## Limitations

The software was tested on Ubuntu 18.04 LTS and Python version >=3.6.x.

## Acknowledgmnets

If you found the software useful please cite:
* Marek Wodzinski and Henning Müller, *DeepHistReg: Unsupervised Deep Learning Registration Framework for Differently Stained Histology Samples*. In preparation. 
The article presents the whole DeepHistReg framework with deep segmentation, initial alignment, affine registration and improved deformable registration.
* Marek Wodzinski and Henning Müller, *Unsupervised Learning-based Nonrigid Registration of High Resolution Histology Images*, 11th International Workshop on Machine Learning in Medical Imaging (MICCAI-MLMI), 2020. 
The article introduces the first version of the nonrigid registration.
* Marek Wodzinski and Henning Müller, *Learning-based Affine Registration of Histological Images*,  9th International Workshop on Biomedical Image Registration (WBIR), 2020.
The article introduces the patch-based, resolution-independent affine registration being a part of the framework.

You may also find useful to cite (however, the DeepHistReg is not part of the challenge summary article):
* J. Borovec *et.al*., *ANHIR: Automatic Non-rigid Histological Image Registration Challenge*, IEEE Transactions on Medical Imaging, 2020, DOI: 10.1109/TMI.2020.2986331

