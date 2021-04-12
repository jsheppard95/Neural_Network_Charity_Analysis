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

## Results
### Data Preprocessing
We first preprocess our data set
[`charity_data.csv`](Resources/charity_data.csv) by reading our data and
identifying the following target, feature, and identification variables:

- Target Variable: `IS_SUCCESSFUL`
- Feature Variables: `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `STATUS`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, `ASK_AMT`
- Identifcation Variables (removed): `EIN`, `NAME`

We then encode categorical variables using
`sklearn.preprocessing.OneHotEncoder` after bucketing noisy features
`APPLICATION_TYPE` and `CLASSIFICATION` with many unique values. After one hot
encoding, we split our data into the target and features, split further into
training and testing sets, and scale our training and testing feature data
using `sklearn.preprocessing.StandardScaler`.

### Compiling, Training, and Evaluating the Model
With our data now preprocessed, we initially build the model defined in
[`AlphabetSoupCharity.ipynb`](AlphabetSoupCharity.ipynb) using
`tensorflow.keras.models.Sequential` and `tensorflow.keras.layers.Dense`
with the following parameters:

| Parameter | Value | Justification |
| --------- | ----- | ------------- |
| Number of Hidden Layers | 2 | Deep neural network is necessary for complex data, good starting point with low computation time. |
| Architecture (hidden_nodes1, hidden_nodes2, ...) | (80, 30) | First layer has roughly two times the number of inputs (43), smaller second layer provides shorter computation time. |
| Hidden Layer Activation Function | `relu` | Simple choice for inexpensive training with generally good performance. |
| Number of Output Nodes | 1 | Model is a binary classifier and should therfore have one output predicting if `IS_SUCCESSFUL` is `True` or `False`. |
| Output Layer Activation Function | `sigmoid` | Provides a probability output (value between 0 and 1) for the classification of `IS_SUCCESSFUL`. |

This yields the model summary shown in
[Base Model Summary](Images/base_model_summary.png). We then compile the model
using the `binary_crossentropy` loss function, `adam` optimizer, and
`accuracy` to obtain the training results shown in
[Base Model Training](Images/base_model_training.png). Verifying with the
testing set, we obtain the following results:

- Loss: 0.559
- Accuracy: 0.729

We next optimize the previous model by tweaking the parameters shown above
and more in
[`AlphabetSoupCharity_Optimization.ipynb`](AlphabetSoupCharity_Optimization.ipynb),
initially making the following single changes to our initial model:

| Parameter | Change | Justification | Loss | Accuracy |
| --------- | ------ | ------------- | ---- | -------- |
| Training Duration (in epochs) | Increase from 100 to 200 | Longer training time could result and more trends learned and greater performance. | 0.582 | 0.728 |
| Hidden Layer Activation Function | Change from `relu` to `tanh` | Scaled data results in negative inputs which `tanh` does not output as zero. | 0.557 | 0.734 |
| Number of Input Features | Reduce from 43 to 34 by bucketing `INCOME_AMT` and `AFFILIATION` and droping the redundant column `SPECIAL_CONSIDERATIONS_N` after encoding. | Less noise in input data. | 0.555 | 0.729 |

We thus do not see a significant increase in performance from the initial
model and do not meet the target 75% accuracy critera. To help remedy this, we
attempt or more systematic approach by following
[Optimizing Neural Networks - Where to Start?](https://towardsdatascience.com/optimizing-neural-networks-where-to-start-5a2ed38c8345)
and iteratively changing a single model parameter while holding others fixed,
and then combining the parameters which generated the highest accuracy in each
iterative search. This results in the following:

| Parameter | Search Options | Optimal Value | Loss | Accuracy |
| --------- | -------------- | ------------- | ---- | -------- |
| Training Duration (in epochs) | [50, 100, 200, 300] | 100 | 0.588 | 0.732 |
| Architecture | All permutations with one to four hidden layers with 10, 30, 50, and 80 nodes, i.e [(10,), ..., (80,), (10, 30), (30, 10), ..., (80, 50), (10, 30, 50), (10, 50, 30), (30, 10, 50), ..., (80, 50, 30), (10, 30, 50, 80), (10, 30, 80, 50), (10, 50, 30, 80), (10, 50, 80, 30), ..., (80, 50, 30, 10)] | (80, 50, 30), i.e three hidden with 80, 50, and 30 nodes. | 0.561 | 0.740 
| Hidden Layer Activation Function | `["relu", "tanh", "selu", "elu", "exponential"]` | `tanh` | 0.556 | 0.734 |
| Number of Input Features | Bucket all combinations of `APPLICATION_TYPE`, `CLASSIFICATION`, `INCOME_AMT`, and `AFFILIATION`, similar options as architecture | Bucket `CLASSIFICATION` only (still drop redundant `SPECIAL_CONSIDERATIONS_N`) resulting in 50 input features. | 0.560 | 0.737 |
| Learning Rate | Coarse search [0.0001, 0.001, 0.01, 0.1, 1], fine search of six random values between 0.0001 and 0.01 | 0.000594 | 0.546 | 0.737 |

Combining all optimized model parameters, we retrain and trest and obtain the
following loss and accuracy:

- Loss: 0.564
- Accuracy: 0.728

We thus find a negligible decrease in accuracy from the base model defined in
[`AlphabetSoupCharity.ipynb`](AlphabetSoupCharity.ipynb). As an additional
attempt at optimization, we perform an iterative search of training
`batch_size` choosing values `[1, 2, 4, 8, 16, 32, 64]` and keeping the above
paramters found through their resepective searches. This search shows that a batch
size of 16 yields the best results with a loss of 0.547 and accuracy of 0.737.

Considering the testing accuracy of each model tested, we find the
architecture search generated the model with highest testing accuracy and had
the following parameters:

| Parameter | Value |
| --------- | ----- |
| Number of Hidden Layers | 3 |
| Architecture (hidden_nodes1, hidden_nodes2, hidden_nodes3) | (80, 50, 30) |
| Hidden Layer Activation Function | `relu` |
| Number of Output Nodes | 1 |
| Output Layer Activation Function | `sigmoid` |
| Learning Rate | 0.001 (default) |
| Training Duration (in epochs) | 100 |
| Bucket Categorical Variables | No |
| Batch Size | 32 |

Rebuilding and training this model, we obtain the summary shown in
[Optimized Model Summary](Images/optimized_model_summary.png) and training
results shown in
[Optimized Model Training](Images/optimized_model_training.png). While in this
case we continue to see promising training accuracy reaching 0.745, we find
the model performance has decreased slightly when faced with the testing data:

- Loss: 0.583
- Accuracy: 0.727