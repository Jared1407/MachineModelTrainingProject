#!/usr/bin/env python3
'''
### OVERVIEW ###
Test: Reads test.csv which will have the Nx24 matrix and outputs a Result.csv file with the Nx1 vector
Result.csv: Nx1 Vector with 1 for meal and 0 for no meal
'''

import pickle
import pandas as pd
from train import FeatureExtraction

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Read test data
data = pd.read_csv('test.csv', low_memory=False, header=None)

# Extract features
features, placeholder = FeatureExtraction(data, data)

# Predict the data
result = model.predict(placeholder)

# Save the result
pd.DataFrame(result).to_csv('Result.csv', index=False, header=False)

