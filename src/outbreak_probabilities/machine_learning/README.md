*This folder contains machine learning models and scripts used to predict outbreak probabilities based on simulated case data generated from serial interval distributions and weights*.

*The models are trained using different numbers of weeks of case data (2 weeks, 3 weeks, 4 weeks, and 5 weeks) to evaluate their performance in predicting the probability of an outbreak.*

## Each script follows a similar structure 
1. Import necessary libraries and modules.
2. Define machine learning models to be used
3. Set paths for data, model storage, and output plots.
4. Load and preprocess the simulated case data.
5. Split the data into training and testing sets.
6. Scale the features using StandardScaler.
7. Train the model on the full dataset and save it in model_*weeks

## Example Usage
 PYTHONPATH=src python -m outbreak_probabilities.predict --list
 
  PYTHONPATH=src python -m outbreak_probabilities.predict --weeks 2 --model RF --week 2.1 --week 1.4
  
  PYTHONPATH=src python -m outbreak_probabilities.predict --weeks 3 --model RF --batch inputs.csv --out preds.csv


