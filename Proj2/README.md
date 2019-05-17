# Project 2: Mini deep-learning framework
Deep Learning [EE-559], 2019, EPFL

This is the directory for Project 2 of the course "Deep Learning" spring 2019, EPFL. This file contains practical information on the project implementation and how to run it. For more detailed explanation of the project (goals, implemented algorithms, ...), please refer to the report (`report.pdf`). 

The objective of this project is to design a mini “deep learning framework” using only pytorch’s tensor operations and the standard math library, hence in particular **without using autograd or the neural-network modules**.

## Getting Started

These instructions contain explanations of the structure of the code and how to get a running version to test the implementation. 


## Project Structure

The project has the following folder (and file) structure:

* `src/`. Folder containing the actual code files for the project:
    * `output/`. Directory containing plots generated when executing `test.py`
    
    * `Activation.py` File containing the definition of the different activation classes.
    * `Linear.py` File containing the definition of the fully connected layer class.
    * `Loss.py` File containing the definition of the different loss classes.
    * `Module.py` File containing the definition of the class Module.
    * `Optimizers.py` File containing the definition of the different optimizer classes.
    * `Sequential.py` File containing the definition of the Sequential class.
    * `data_generation.py` File containing different functions to generate the data to be used in the project.
    * `evaluate_models.py` File containing different functions to train the models as well as evaluate their performance.
    * `test.py` Runnable file to check the proper functioning of the framework by evaluating different models. 
    

## Main Program

To run the main program, the `test.py` has to be executed from the `\src` directory:
```
cd Proj2/src/
python test.py
``` 

There are no command line arguments. 
The program will output the following progress updates though the standard output. 

```
Run number 1 / 15 :

ATTENTION: PLOTS WILL NOT BE SHOWN.
ALL OF THEM ARE STORED IN THE OUTPUT FOLDER
Generate data

Model 1: Optimizer: SGD; No dropout; ReLU; CrossEntropy
Epochs 950. Loss: xxx
Loss:  xxx
Number of errors:  x

Loss:  xxx
Number of errors:  x

...

Evaluation of different activation functions

...

Evaluation of base model with MSE loss

...

Loss:  xxx
Number of errors:  x
Evaluation done!


Run number 2 / 15 :

...
...

Run number 15 / 15 :

...

Evaluation done!

```

Where for both the training and test of the models, a percentage of the completed training is shown for each run and each model. The final loss and number of errors in the train and test datasets is also provided.

Additionally, the program generates different plots that are stored in `../output`

The whole program may take around **25 minutes to complete the execution**. 


## Authors

* **Bolón Brun, Natalie** - 
* **Lemaitre, Eugène** - 
* **Munier, Louis** - 






