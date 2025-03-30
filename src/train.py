#!/usr/bin/env python3
'''
### OVERVIEW ###

Train: Reads CGMData.csv, CGM_patient2.csv, InsulinData.csv and Insulin_patient2.csv
Extracts meal and no meal data, extracts features,
Trains the model to recognize meal and no meal classes and store to model in a pickle file
Python API pickle
Assumptions:
1:  CGMData.csv, CGM_patient2.csv, InsulinData.csv, Insulin_patient2.csv, and test.csv files are all in your compilation and execution folder.

Technology Requirements:
scikit-learn 1.3.2
pandas 1.5.3
numpy 1.26.3
scipy 1.11.4

Important Data from InsulinData.csv:
Date time in which meal was taken (date/time at BWZ Carb Input (grams)!= 0)
    From this date and time stamp, Extract the CGM data from CGMData.csv and CGM_patient2.csv in a 2hour period AFTER the date and 30 minutes BEFORE
Important Data from CGMData.csv:
Blood Glucose Level (mg/dL) from the above time period

Data Processing:

Take all the glucose values from 30 minutes before tm and 2 hours after and create a meal data matrix
Meal Data Matrix (Px30):
    time tm will be the time of the meal. We will only use the data 30 mins before and 2 hours after tm iff:
    1: There is no meal from time tm to tm + 2 hours
    2: If there is a meal from tm to tm + 2 hrs, we will begin tm at the latest meal within the 2 hour period
    3: If there is a meal at tm+2hrs, consider tm+1.5hrs to tm+4hrs a meal data point
    # tm - 30 mins to tm + 2 hours
    Col: 30 max (2.5 hours at 5 min intervals)
    Row: 1 Row per meal data point P
    Data: Blood Glucose Level (mg/dL)
    P: Number of meal data time series

No Meal Data Matrix (Qx24):
    Anything after the tm + 2 hours for 2 hours ie tm + 2 hours to tm + 4 hours
    We will call this post-absorptive period (tm + 2 hours to tm + 4 hours)
    Conditions:
    1: There is no meal found in the post-absorptive period, if there is we cannot consider this as a no meal data point
    2: If the meal value found in the post-absorptive period is 0, ignore this and consider the strech
    Col: 24 max (2 hours at 5 min intervals)
    Row: 1 Row per no meal data point Q
    Data: Blood Glucose Level (mg/dL)
    Q: Number of no meal data time series

There will be missing data:
If there is an amount greater than x missing data points in a given row, we can ignore that row
If there is an amount greater than y missing data points in a given row, we can use polynomial regression to fill in the missing data points
x and y will be determined from great amounts of testing
We could also use KNN to fill in the missing data points, needs testing

Feature Extraction:
    Now that we have meal data (Px30 matrix) and no meal data (Qx24 matrix), we can extract features
    Features: The meal data will have specific features when plotting CGM data over time
        Tao: Time between tm and the peak glucose value after tm and before tm + 2 hours
        DGnormalized: (Max CGM Value - Meal CGM Value)/ Meal CGM Value
        Fast Fourier Transform: Plot power vs frequency and extract the power and frequency at the 2nd and 3rd peak given a 4 tuple (frequency, power) for each peak
    We can have more features extracted but for now we will stick to these 3 giving us 6 features
    Feature Lenght (Fl) = 6
    These features will be stored in a PxFl matrix. 6 features per meal data point
    This will be called the Meal Feature Matrix
    We will also have a QxFl matrix for the no meal data points
    This will be the No Meal Feature Matrix
    Now both features will be uniform in column size.

Model Training:
    Now that we have 2 Feature Matrices, of PxFl and QxFl we will concatenate them to form a (P+Q)xFl matrix
    This will be the training data
    [PxFl] = [(P+Q)xFl]
    [QxFl]
    We will need a label vector of (P+Q)x1, this will consist of P 1's and Q 0's
    Now we have the training data and the label vector
    We can now train the model using either an SVM or Decision Tree Classifier
    Split Data: 80% Training, 20% Testing:
    (80/100P + 80/100Q) x Fl = 80/100(P+Q)xFl with [80/100P, 80/100Q] = Training Data
    (20/100P + 20/100Q) x Fl = 20/100(P+Q)xFl with [20/100P, 20/100Q] = Testing Data

Model Validation:
    Precision, Recall, F1 Score, Confusion Matrix

Model Storage:
    Store the model in a pickle file to be used by the test.py file on test.csv
'''
# Import Libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import pickle


# Data Processing
def DataProcessing(cgm_data, ins_data):
    # Read CGMData.csv, CGM_patient2.csv, InsulinData.csv and Insulin_patient2.csv
    # Extract meal and no meal data

    # Read InsulinData.csv to identify meal times
    # Search the BWZ Carb Input (grams) column for non-zero values
    # Once found, loop through cgm data until we find a date that is greater than or equal to the time
    # We take the index at this position and count the next 24 points as meal data as well as the previous 6 points

    # Psuedo code:
    '''
    Loop through ins_data until BWZ Carb Input (grams) != 0, the date at this location will be the meal date
    Continue through ins_data until we find the next date and call this new date
    if new date is within 2 hours of meal date, new date will be the meal date. Repeat until new date is not within 2 hours
    
    Search through cgm_data for the meal date (meal_date >= cgm_data['Date'] + cgm_data['Time'])
    Once the date is found we take the next 24 points and the previous 6 points as meal data using Index
    We will store this data in a matrix
    '''

    meal_times = []

    for n in ins_data.index:
        if ins_data.loc[n, 'BWZ Carb Input (grams)'] >= 0.00000001:
            meal_datetime = ins_data.loc[n, 'DateTime']
            for x in range(1,20):
                y = n - x
                if ins_data.loc[y, 'BWZ Carb Input (grams)'] >= 0.00000001:
                    new_datetime = ins_data.loc[y, 'DateTime']
                    # If new date is within 2 hours of meal date, new date will be the meal date
                    if int((new_datetime - meal_datetime).seconds) < int(pd.Timedelta(hours=2).seconds):
                        n = y
                        meal_datetime = new_datetime
                    elif int((new_datetime - meal_datetime).seconds) == int(pd.Timedelta(hours=2).seconds):
                        # If there is a meal at time mealtime + 2hrs exactly, consider tm+1.5hrs to tm+4hrs a meal data point
                        # This can be simplified to mean take mealtime + 1.5 hours as the new mealtime
                        n = y
                        new_datetime = meal_datetime + pd.Timedelta(hours=1.5)
                        meal_datetime = new_datetime
            meal_times.append(meal_datetime)
    # Remove duplicates (Im still unsure why there are duplicates)
    meal_times = list(dict.fromkeys(meal_times))

    # Create a list of times that are outside the meal times (X < tm, tm + 2 < Y)
    # Proper no meal data added:
    no_meal_times = []
    first_no_meal_data = ins_data.loc[len(ins_data)-1, 'DateTime']
    no_meal_times.append(first_no_meal_data)
    copy_meal_times = meal_times.copy()
    for n in ins_data.index:
        x = copy_meal_times[0]
        y = copy_meal_times[1]
        if ins_data.loc[n, 'DateTime'] >= y:
            copy_meal_times.pop(0)
            if len(copy_meal_times) <= 1:
                break
            continue
        if (x + pd.Timedelta(hours=2)) <= ins_data.loc[n, 'DateTime'] < y:

            # Only add a non meal time if it is also 2 hours or more after the last none meal time
            #
            if (ins_data.loc[n, 'DateTime']) >= (no_meal_times[len(no_meal_times)-1] + pd.Timedelta(hours=2)):
                no_meal_time = ins_data.loc[n, 'DateTime']
                no_meal_times.append(no_meal_time)


    # Now we have the datetime of the meal, we need to find the index of the nearest date in the cgm_data
    #print(meal_datetime, ins_data.loc[n, 'BWZ Carb Input (grams)'])
    meal_data = np.zeros((len(meal_times), 30))
    # Make a copy of the meal_times list for the no meal data
    for m in cgm_data.index:
        cgm_datetime = cgm_data.loc[m, 'DateTime']
        if cgm_datetime >= meal_times[0]:
            for x in range(29):
                try:
                    y = m + x - 6
                    if float(cgm_data.loc[y, 'Sensor Glucose (mg/dL)']) == 0:
                        meal_data[len(meal_times)][x] = 5
                    else:
                        meal_data[len(meal_times)][x] = float(cgm_data.loc[y, 'Sensor Glucose (mg/dL)'])
                except:
                    continue
            #print(cgm_datetime, meal_times[len(meal_times)-1])
            if len(meal_times) > 1:
                meal_times.pop()
            else:
                break

    # Really this should be done differently, but for now we will just use the same method
    # It should search for from one meal to the next, and for all the data in between after 2 hours of the meal, split it into 24 points
    # and each of these 24 points will be a no meal data point
    no_meal_data = np.zeros((len(no_meal_times), 24))
    for m in cgm_data.index:
        cgm_datetime = cgm_data.loc[m, 'DateTime']
        post_meal_datetime = no_meal_times[0]
        if cgm_datetime >= post_meal_datetime:
            for x in range(23):
                try:
                    y = m + x
                    if float(cgm_data.loc[y, 'Sensor Glucose (mg/dL)']) == 0:
                        no_meal_data[len(meal_times)][x] = 0
                    else:
                        no_meal_data[len(meal_times)][x] = float(cgm_data.loc[y, 'Sensor Glucose (mg/dL)'])
                except:
                    continue
            if len(no_meal_times) > 1:
                no_meal_times.pop()
            else:
                break

    meal_data = np.array(meal_data)
    no_meal_data = np.array(no_meal_data)

    return meal_data, no_meal_data


# Feature Extraction
def FeatureExtraction(meal_data, no_meal_data):
    # We will extract the features from both meal and no meal data
    # We will create a PxFl matrix for meal data and a QxFl matrix for no meal data
    # Fl = 6, Tao, DGnormalized, FFT 2nd peak frequency, FFT 2nd peak power, FFT 3rd peak frequency, FFT 3rd peak power

    # Begin with meal data

    # Tao: Time between tm and the peak glucose value after tm and before tm + 2 hours
    # DGnormalized: (Max CGM Value - Meal CGM Value)/ Meal CGM Value
    # Fast Fourier Transform: Plot power vs frequency and extract the power and frequency at the 2nd and 3rd peak given a 4 tuple (frequency, power) for each peak

    # Accuracy was low so we need to add features:
    # We should take the first and second differential of the cgm data and use this as a feature
    # Differential with respect to time: Dcgm/Dt(D/Dt)
    # We should also take the max value of the cgm data and use this as a feature
    # We should find the greatest slope value and use that as a feature as well


    # We will use the numpy library to extract the features

    # drop nan values
    meal_data = meal_data[~np.isnan(meal_data).any(axis=1)]
    no_meal_data = no_meal_data[~np.isnan(no_meal_data).any(axis=1)]

    # Create a scalar object to normalize the data
    scalar = preprocessing.StandardScaler()
    meal_data = scalar.fit_transform(meal_data)
    no_meal_data = scalar.fit_transform(no_meal_data)

    meal_feature_array = []
    for x in meal_data:
        # Find maximum slope of CGM vs Time(Data Points)
        slope = max(np.diff(x))
        second_diff = max(np.diff(np.diff(x)))/slope

        # Standard Deviation:
        std_dev = np.std(x)

        # Tao
        meal_cgm_value = x[0]
        max_cgm_value = np.max(x)
        tao = np.argmax(x) * 5

        # DG Normalized
        dg_normalized = (max_cgm_value - meal_cgm_value) / meal_cgm_value

        # FFT
        fft_data = np.fft.fft(x)
        power = np.abs(fft_data)
        freq = np.fft.fftfreq(len(x), d=5 / 60)

        peaks = np.argsort(power)[-3:-1]
        peak_freqs = freq[peaks]
        peak_powers = power[peaks]

        meal_feature_array.append([tao, dg_normalized, peak_freqs[0], peak_powers[0], peak_freqs[1], peak_powers[1], second_diff, std_dev])

    meal_feature_matrix = np.array(meal_feature_array)

    # Now we will do the same for no meal data
    no_meal_feature_array = []

    for x in no_meal_data:
        # Find maximum slope of CGM vs Time(Data Points)
        slope = max(np.diff(x))
        second_diff = max(np.diff(np.diff(x))) / slope

        # Standard Deviation:
        std_dev = np.std(x)

        # Tao
        meal_cgm_value = x[0]
        max_cgm_value = np.max(x)
        tao = np.argmax(x) * 5

        # DG Normalized
        dg_normalized = (max_cgm_value - meal_cgm_value) / meal_cgm_value

        # FFT
        fft_data = np.fft.fft(x)
        power = np.abs(fft_data)
        freq = np.fft.fftfreq(len(x), d=5 / 60)

        peaks = np.argsort(power)[-3:-1]
        peak_freqs = freq[peaks]
        peak_powers = power[peaks]

        no_meal_feature_array.append([tao, dg_normalized, peak_freqs[0], peak_powers[0], peak_freqs[1], peak_powers[1], second_diff, std_dev])

    no_meal_feature_matrix = np.array(no_meal_feature_array)


    return meal_feature_matrix, no_meal_feature_matrix

# Model Training
def ModelTraining(meal_feature_matrix, no_meal_feature_matrix):
    # Concatenate the matrices to form a training data matrix
    X = np.concatenate((meal_feature_matrix, no_meal_feature_matrix))

    # Create a label vector as follows: P 1's and Q 0's
    y = np.concatenate((np.ones(len(meal_feature_matrix)), np.zeros(len(no_meal_feature_matrix))))

    # Split the data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest model
    rand_forest = RandomForestClassifier(max_depth=4, random_state=42)
    rand_forest.fit(X,y)

    # Train the model (SVM first)
    svm = SVC(kernel='rbf', C=1, gamma='scale')
    svm.fit(X_train, y_train)

    # Train Linear SVC 
    svc = make_pipeline(preprocessing.StandardScaler(), LinearSVC())
    svc.fit(X_train, y_train)

    return svc, X_test, y_test

# Model Validation
def ModelValidation(svm, X_test, y_test):
    # Model Validation
    y_pred = svm.predict(X_test)
    print("Model Validation")
    print("Precision: ", precision_score(y_test, y_pred))
    print("Recall: ", recall_score(y_test, y_pred))
    print("F1 Score: ", f1_score(y_test, y_pred))
    print("Confusion Matrix: ")
    print(confusion_matrix(y_test, y_pred))


def main():
    # Use pandas to read in csv files to dataframes
    ins_input = pd.read_csv('InsulinData.csv', low_memory=False,
                            usecols=['Date', 'Time', 'BWZ Carb Input (grams)'])
    cgm_input = pd.read_csv('CGMData.csv', low_memory=False,
                            usecols=['Date', 'Time', 'Sensor Glucose (mg/dL)'])
    ins_input_patient2 = pd.read_csv('Insulin_patient2.csv', low_memory=False,
                                     usecols=['Date', 'Time', 'BWZ Carb Input (grams)'])
    cgm_input_patient2 = pd.read_csv('CGM_patient2.csv', low_memory=False,
                                     usecols=['Date', 'Time', 'Sensor Glucose (mg/dL)'])

    # We should create a datetime column for the date and time as writting date + time is not too efficient
    # For some godforsaken reason, the date column in ins_input_patient2 will randomly add 00:00:00 to the end of the date
    # So we need to remove this
    ins_input_patient2['Date'] = ins_input_patient2['Date'].str.replace(' 00:00:00', '')
    ins_input['DateTime'] = pd.to_datetime(ins_input['Date'] + ' ' + ins_input['Time'])
    cgm_input['DateTime'] = pd.to_datetime(cgm_input['Date'] + ' ' + cgm_input['Time'])
    ins_input_patient2['DateTime'] = pd.to_datetime(ins_input_patient2['Date'] + ' ' + ins_input_patient2['Time'])
    cgm_input_patient2['DateTime'] = pd.to_datetime(cgm_input_patient2['Date'] + ' ' + cgm_input_patient2['Time'])

    # The 2 patients files were in different orders/ can be in different orders so we need to sort them:
    ins_input = ins_input.sort_values(by='DateTime')
    cgm_input = cgm_input.sort_values(by='DateTime')
    ins_input_patient2 = ins_input_patient2.sort_values(by='DateTime')
    cgm_input_patient2 = cgm_input_patient2.sort_values(by='DateTime')

    # Process the data
    meal_data, no_meal_data = DataProcessing(cgm_input, ins_input)
    meal_data_patient2, no_meal_data_patient2 = DataProcessing(cgm_input_patient2, ins_input_patient2)

    # We now have meal data as a Px30 matrix and no meal data as a Qx24 matrix for both patients (Ppat1x30, Qpat1x24, Ppat2x30, Qpat2x24)
    # We can now combine these matrices to form a Px30 + Ppat2x30 and Qx24 + Qpat2x24
    # We can then extract features from this data
    meal_data_combined = np.concatenate((meal_data, meal_data_patient2), axis=0)
    no_meal_data_combined = np.concatenate((no_meal_data, no_meal_data_patient2), axis=0)

    # Extract features
    meal_feature_matrix, no_meal_feature_matrix = FeatureExtraction(meal_data_combined, no_meal_data_combined)

    # Train the model
    svm, X_test, y_test = ModelTraining(meal_feature_matrix, no_meal_feature_matrix)

    # Model Validation
    ModelValidation(svm, X_test, y_test)

    # Save the model to a pickle file
    with open('model.pkl', 'wb') as f:
        pickle.dump(svm, f)

if __name__ == '__main__':
    main()
