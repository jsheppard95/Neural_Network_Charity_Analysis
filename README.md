# Neural_Network_Charity_Analysis
Neural network binary classifier to predict if charities will be successful if
they receive funding. 

## Overview of the analysis
This project includes Jupyter Notebook files that build, train, test, and
optimize a deep neural network that models charity success from nine features
in a loan application data set. We employ the TensorFlow Keras `Sequential`
model with `Dense` hidden layers and a binary classification output layer and
optimize this model by varying the following parameters:

- Training duration (in epochs)
- Hidden layer activation functions
- Hidden layer architecture
- Number of input features through categorical variable bucketing
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
noting the following target, feature, and identification variables:

- Target Variable: `IS_SUCCESSFUL`
- Feature Variables: `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `STATUS`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, `ASK_AMT`
- Identification Variables (to be removed): `EIN`, `NAME`

We then encode categorical variables using
`sklearn.preprocessing.OneHotEncoder` after bucketing noisy features
`APPLICATION_TYPE` and `CLASSIFICATION` with many unique values. After one hot
encoding, we split our data into the target and features, split further into
training and testing sets, and scale our training and testing data using
`sklearn.preprocessing.StandardScaler`.

### Compiling, Training, and Evaluating the Model
With our data preprocessed, we build the base model defined in
[`AlphabetSoupCharity.ipynb`](AlphabetSoupCharity.ipynb) using
`tensorflow.keras.models.Sequential` and `tensorflow.keras.layers.Dense`
with the following parameters:

| Parameter | Value | Justification |
| --------- | ----- | ------------- |
| Number of Hidden Layers | 2 | Deep neural network is necessary for complex data, good starting point with low computation time. |
| Architecture (hidden_nodes1, hidden_nodes2) | (80, 30) | First layer has roughly two times the number of inputs (43), smaller second layer offers shorter computation time. |
| Hidden Layer Activation Function | `relu` | Simple choice for inexpensive training with generally good performance. |
| Number of Output Nodes | 1 | Model is a binary classifier and should therefore have one output predicting if `IS_SUCCESSFUL` is `True` or `False`. |
| Output Layer Activation Function | `sigmoid` | Provides a probability output (value between 0 and 1) for the classification of `IS_SUCCESSFUL`. |

This yields the model summary shown in
[Base Model Summary](Images/base_model_summary.png). We then compile and train
the model using the `binary_crossentropy` loss function, `adam` optimizer, and
`accuracy` metric to obtain the training results shown in
[Base Model Training](Images/base_model_training.png). Verifying with the
testing set, we obtain the following results:

- Loss: 0.559
- Accuracy: 0.729

We next optimize the previous model by adjusting the parameters shown above
and more in
[`AlphabetSoupCharity_Optimization.ipynb`](AlphabetSoupCharity_Optimization.ipynb),
initially making the following single changes:

| Parameter | Change | Justification | Loss | Accuracy |
| --------- | ------ | ------------- | ---- | -------- |
| Training Duration (epochs) | Increase from 100 to 200 | Longer training time could result in more trends learned. | 0.582 | 0.728 |
| Hidden Layer Activation Function | Change from `relu` to `tanh` | Scaled data results in negative inputs which `tanh` does not output as zero. | 0.557 | 0.734 |
| Number of Input Features | Reduce from 43 to 34 by bucketing `INCOME_AMT` and `AFFILIATION` and dropping the redundant column `SPECIAL_CONSIDERATIONS_N` after encoding. | Less noise in the input data. | 0.555 | 0.729 |

We thus do not see a significant increase in performance from the initial
model and do not meet the target 75% accuracy criteria. To remedy this, we
attempt a systematic approach by following
[Optimizing Neural Networks - Where to Start?](https://towardsdatascience.com/optimizing-neural-networks-where-to-start-5a2ed38c8345)
and iteratively changing one model parameter at a time while holding others
fixed, and then combining the parameters which generated the highest accuracy
in each iterative search. This results in the following:

| Parameter | Search Options | Optimal Value | Loss | Accuracy |
| --------- | -------------- | ------------- | ---- | -------- |
| Training Duration (epochs) | [50, 100, 200, 300] | 100 | 0.588 | 0.732 |
| Architecture | All permutations with one to four hidden layers with 10, 30, 50, and 80 nodes, i.e [(10,), ..., (80,), (10, 30), (30, 10), ..., (80, 50), (10, 30, 50), (10, 50, 30), (30, 10, 50), ..., (80, 50, 30), (10, 30, 50, 80), (10, 30, 80, 50), (10, 50, 30, 80), ..., (80, 50, 30, 10)] | (80, 50, 30), i.e three hidden layers with 80, 50, and 30 nodes. | 0.561 | 0.740 |
| Hidden Layer Activation Function | [`relu`, `tanh`, `selu`, `elu`, `exponential`] | `tanh` | 0.556 | 0.734 |
| Number of Input Features | Bucket all combinations of `APPLICATION_TYPE`, `CLASSIFICATION`, `INCOME_AMT`, and `AFFILIATION`, similar structure as Architecture | Bucket `CLASSIFICATION` only (still drop redundant `SPECIAL_CONSIDERATIONS_N`) resulting in 50 input features. | 0.560 | 0.737 |
| Learning Rate | Coarse search [0.0001, 0.001, 0.01, 0.1, 1], fine search of six random values between 0.0001 and 0.01 | 0.000594 | 0.546 | 0.737 |

Combining all optimized model parameters, we retrain and test to obtain the
following testing loss and accuracy:

- Loss: 0.564
- Accuracy: 0.728

We thus find a negligible decrease in accuracy from the base model defined in
[`AlphabetSoupCharity.ipynb`](AlphabetSoupCharity.ipynb). As an additional
optimization attempt, we perform an iterative search of training batch size
with values `[1, 2, 4, 8, 16, 32, 64]` and retaining the previously
optimized parameters. This search shows that a batch size of 16 yields the
best results with a loss of 0.547 and accuracy of 0.737.

Considering the testing accuracy of each model, we find the architecture
search generated the model with the best results and had the following
parameters:

| Parameter | Value |
| --------- | ----- |
| Number of Hidden Layers | 3 |
| Architecture (hidden_nodes1, hidden_nodes2, hidden_nodes3) | (80, 50, 30) |
| Hidden Layer Activation Function | `relu` |
| Number of Output Nodes | 1 |
| Output Layer Activation Function | `sigmoid` |
| Learning Rate | 0.001 (default) |
| Training Duration (epochs) | 100 |
| Bucket Categorical Variables | No |
| Batch Size | 32 (default) |

Rebuilding and training this model, we obtain the summary shown in
[Optimized Model Summary](Images/optimized_model_summary.png) and results
shown in [Optimized Model Training](Images/optimized_model_training.png).
While in this case we see the training accuracy reaches a promising 0.745, we
find the model performance has decreased slightly when faced with the testing
data:

- Loss: 0.583
- Accuracy: 0.727

## Summary
In summary, we present a deep neural network classification model that
predicts loan applicant success from feature data contained in
[`charity_data.csv`](Resources/charity_data.csv) with 73% accuracy. This does
not meet the 75% accuracy target, and the optimization methods employed here
have not caused significant improvement.

### Additional Optimization Methods
Performance could increase through additional optimization techniques such as
visualizing the numerical feature variable `ASK_AMT` to find and remove
potential outliers that could be causing noise. Additionally, one could
iteratively tune the parameters above and keep optimal values when moving to
subsequent parameters instead of reverting to the base setting and combining
after completion. This would however require more careful thought on the order
with which one adjusts parameters to arrive at an optimized model.

### Alternative Models
An alternative to the deep learning classification model presented in this
project could be a more traditional Random Forest Classifier. This model is
also appropriate for this binary classification problem and can often perform
comparably to deep learning models with just two hidden layers. It is also
advantageous in that there are less parameters to optimize and those which do
require attention are more intuitive than those in a neural network.

## Usage
All code is contained in the Jupyter Notebook files
[`AlphabetSoupCharity.ipynb`](AlphabetSoupCharity.ipynb) and
[`AlphabetSoupCharity_Optimization.ipynb`](AlphabetSoupCharity_Optimization.ipynb).
Therefore to replicate the results of this analysis, clone this repository and
install the necessary dependencies into an isolated `conda` environment using
the command:
```
conda env create -f environment.yml
```
On can then build, train, and test the classification model with baseline
parameters by opening `AlphabetSoupCharity.ipynb` and running all cells.
The user can then optimize this model by opening
`AlphabetSoupCharity_Optimization.ipynb` and either running all cells
(warning: the architecture and categorical bucketing iterative searches
complete in roughly one hour), or by using the function `build_train_test`
to perform additional iterative searches for optimal parameters with the
following structure:
```
In [1]: learning_rates = [0.0001, 0.001, 0.01, 0.1, 1]                          

In [2]: results = []                                                            

In [3]: for rate in learning_rates: 
   ...:     result = build_train_test(learning_rate=rate, architecture=(80, 30), 
   ...:                               activation="relu", epochs=100, 
   ...:                               cat_cutoffs={"CLASSIFICATION": 1800}, 
   ...:                               batch_size=32) 
   ...:     results.append(result)
```
Here the default values were passed to parameters other than learning rate for
clarity. The parameter `architecture` is a tuple whose length specifies the
number of hidden layers and values the number of nodes in each layer. In this
example there are two hidden layers, the first with 80 nodes and the second
with 30. The parameter `cat_cutoffs` is a dictionary with keys specifying
which categorical features should have bucketing and values the minimum number
of sample value occurences to stay out of the bucket. In this example, if a
sample's value in `CLASSIFICATION` occurs less than 1800 times, its value is
changed to `OTHER`. This function returns a tuple
`(model_loss, model_accuracy)` which in this example is added to `results` for
later analysis.