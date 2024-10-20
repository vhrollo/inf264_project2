# Project overview
Run the notebook 'main_notebook.ipynb' to reproduce the results in our report.
Some visualizations used report are in the separate notebook 'visualizing_the_data'.
The figures mentioned (but not included) in the report can be found in the 'imgs'.

### Python libraries used:
- python=3.11.10
- jupyter
- matplotlib
- Seaborn
- scikit-learn
- scikit-images
- pytorch
- tensorflow = 2.14.0
- Keras = 2.14.0
- keras-tuner = 1.4.7
- numpy = 1.26.4

The project was implemented using python version 3.11.10.
Note the requiered version on some of the libraries. These must be downloaded to these versions. 
A note here, when downgrading Tensorflow, it can change the NumPy version. After installing tf, ensure that you also have NumPy = 1.26.4.
(other version combinations seem to cause issues between ImageGeneration and keras saving/loading)S

### Project information
The classifiers are implemented in their respective '.py' files. These models take the whole dataset as argument, an does an identical seeded split.
This is done to keep the main file nice and clean, and the seed ensures all models use the same train/val/test split.

The methods used for preprocessing are implemented in 'preprocessing.py'.

The network architecture for both CNN classes is implemented in 'nets.py'.

We include pretrained versions of our CNN models in the 'models' folder. By default, these are loaded.
If you wish to train from scratch, do this in the respective cell:

pytorch:
-

keras:
- Set retrain = True
- uncomment .fit
- set Load = False in .evaluate
