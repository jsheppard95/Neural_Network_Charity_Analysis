# Neural_Network_Charity_Analysis
Neural network binary classifier to predict predict if charities will be
successful if they receive external funding. 

## Overview of the analysis
This project includes Jupuyter Notebook files that build, train, test, and
optimize a deep neural network that models charity success from nine features
in a loan application data set including affiliation, income amount, asking
amount, and more. We employ the TensorFlow Keras `Sequential` model with
`Dense` hidden layers and a binary classification output layer and optimize
this model by varying the following parameterings:

- Training duration (in epochs)
- Hidden layer activation functions
- Hidden layer architecture
- Amount of input feature reduction through categorical bucketing
- Learning rate
- Batch size

### Resources
- Data Source:
    - [`charity_data.csv`](Resources/charity_data.csv)
- Software:
    - Python 3.7.6
    - scikit-learn 0.22.1
    - pandas 1.0.1
    - TensorFlow 2.4.1
    - NumPy 1.19.5
    - Matplotlib 3.1.3
    - Jupyter Notebook 1.0.0
