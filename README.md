# Mental Health Tracker project

# Mental Health Tracker


This project aims to develop a mental health tracker using machine learning algorithms to predict and evaluate mental health outcomes. The project utilizes regression algorithms to build predictive models and assess their performance on both training and testing datasets.

## Dataset

The mental health tracker utilizes a dataset containing features and corresponding mental health outcomes. The dataset is divided into two sets: the training set (xtrain, ytrain) used to train the models, and the testing set (xtest, ytest) used to evaluate the models' performance.

## Regression Algorithms

The following regression algorithms are evaluated in this project:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Support Vector Regression (SVR)
- Random Forest Regression

For each algorithm, a grid search is performed to find the optimal hyperparameters using cross-validation. The evaluation metric used is the root mean squared error (RMSE), and the models are assessed using k-fold cross-validation (k=5).

## Evaluation

The best performing regression algorithm is determined based on the lowest RMSE on the training set. The model with the lowest RMSE is selected as the best regressor and evaluated on the testing set.

The following metrics are used to evaluate the models:

- Mean Squared Error (MSE): Measures the average squared difference between the predicted and actual values.
- RMSE: Represents the square root of the MSE and provides a more interpretable error metric.
- R2 Score: Indicates the proportion of variance in the target variable explained by the model.

## Results

The model performance is assessed separately for the training and testing sets. The results provide insights into how well the selected regressor performs on both datasets.

The model performance for the training set:
- MSE: [MSE value]
- RMSE: [RMSE value]
- R2 Score: [R2 score value]

The model performance for the testing set:
- MSE: [MSE value]
- RMSE: [RMSE value]
- R2 Score: [R2 score value]

Please note that the code is not included in this README file but can be found in the corresponding code files.

For more details and to run the code, please refer to the code files in this repository.

Feel free to reach out if you have any questions or need further assistance.
