# Outbreak Probability Prediction using Machine Learning (RF)
This directory contains code and models for predicting outbreak probabilities using machine learning techniques.

The models are trained on simulation data and can predict the probability of a major outbreak based on weekly case counts

## Getting Starting
1. Data Preparation: Ensure you have the simulation data in CSV format as expected by the training scripts.
2. Training Models: Run the respective training scripts (e.g., ML_2weeks.py, ML_3weeks.py, ML_5weeks.py) to train models on the desired number of weeks of data.
3. Varying Data Sizes: Use ML_different_data_sizes.py to train models on different sizes of training data to observe performance changes.
4. Plotting Results: Use Plot_ML_different_data_sizes.py to visualize how model predictions converge to analytical solutions as training data size increases.
5. Predict.py: Use the predict.py file to make predictions using the trained models (2,3,4,and 5 weeks) via command-line interface.


## Model Prediction
Each training script includes a helper function (predict_pmo) to load the trained model and scaler, and make predictions based on weekly case counts. It outputs the predicted class and label (major, minor) as well as the predicted probability.




