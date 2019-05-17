# Project 1: Classification, weight sharing, auxiliary losses
Deep Learning [EE-559], 2019, EPFL

This is the directory for Project 1 of the course "Deep Learning" spring 2019, EPFL. This file contains practical information on the project implementation and how to run it. For more detailed explanation of the project (goals, implemented algorithms, ...), please refer to the report (`report.pdf`). 

The objective of this project is to test different architectures to compare two digits visible in a two-channel image. It aims at showing in particular the impact of weight sharing, and of the use of an auxiliary loss to help the training of the main objective.
## Getting Started

These instructions contain explanations of the structure of the code and how to get a running version to test the implementation. 


## Project Structure

The project has the following folder (and file) structure:

* `output/`. Directory containing plots generated when executing `test.py`

* `src/`. Folder containing the actual code files for the project:
    * `dlc_practical_prologue.py` Contains helper functions to obtain, load and parse the data.
    * `generate_data.py` Contains helper functions to transform the data (normalize, shuffle, etc.)
    * `graphics.py` Contains different functions to generate plots of the evolution of the loss along the training of different models.
    * `models.py` Contains the definition of the different models compared in the project. 
    * `train.py` Contains the functions used for training the models as well as to evaluate their performance in terms of numbre of errors. 
    * `test.py` Runnable file to check the performance of the different models. 
    

## Main Program

To run the main program, the `test.py` has to be executed from the `\src` directory:
```
cd Proj1/src/
python test.py
``` 

There are no command line arguments. It will look for data files located on `\data`. If these files does not exist, it will download the `MNIST`dataset from http://yann.lecun.com/exdb/mnist/ [Accessed May, 2019]

The program will output the following progress updates though the standard output. 

```
ATTENTION: PLOTS WILL NOT BE SHOWN.
ALL OF THEM ARE STOREM IN THE OUTPUT FOLDER
Generate data
Generate Variables

 --- Evaluate models for hyperparameter tunning ---
Hyperparameter tunning for Model 1.1

...


----Evaluate models with TEST set----

Evaluate model 1.1
...

Model with 70332 parameters
Mean error: 5.90 Std deviation in error: 0.88
DONE!
```

Where for both the training and test of the models, a percentage of the completed training is shown for each run and each model. 

Additionally, the program generates different plots that are stored in `../output`

The whole program may take up to **4 hours to complete the execution**. 


## Authors

* **Bolón Brun, Natalie** - 
* **Lemaitre, Eugène** - 
* **Munier, Louis** - 






