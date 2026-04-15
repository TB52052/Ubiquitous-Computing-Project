'''
import os
print(os.listdir())
print(os.getcwd())
os.chdir("C:/Users/taylo/Documents/UNI/Modules/CMP-6046B-25-SEM2-B - Ubiquitous Computing/Coursework/")
'''

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
import itertools


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import tensorflow as tf


# Index for each activity
activity_indices = {
  'Stationary': 0,
  'Phone-side':1,
  'Walking': 2,
  'Running':3,
  'Sitting-down': 4,
  'Phone-drop': 5,
  'Jumping-stand': 6, 
  'Jumping-down': 7,
  #'Falling-combined':8
  'Falling-back': 8,
  'Falling-feint':9,
  'Falling-front':10
}



def compute_raw_data(dir_name):

  raw_data_features = None
  raw_data_labels = None
  interpolated_timestamps = None

  sessions = set()
  # Categorize files containing different sensor sensor data
  file_dict = dict()
  # List of different activity names
  activities = set()
  file_names = os.listdir(dir_name)

  for file_name in file_names:
    if '.txt' in file_name:
      tokens = file_name.split('-')
      identifier = '-'.join(tokens[4: 6])
      activity = '-'.join(file_name.split('-')[6:-2])
      sensor = tokens[-1]
      sessions.add((identifier, activity))

      if (identifier, activity, sensor) in file_dict:
        file_dict[(identifier, activity, sensor)].append(file_name)
      else:
        file_dict[(identifier, activity, sensor)] = [file_name]


  for session in sessions:
    accel_file = file_dict[(session[0], session[1], 'accel.txt')][0]
    accel_df = pd.read_csv(dir_name + '/' + accel_file)
    accel = accel_df.drop_duplicates(accel_df.columns[0], keep='first').values
    # Spine-line interpolataion for x, y, z values (sampling rate is 32Hz).
    # Remove data in the first and last 3 seconds.
    timestamps = np.arange(accel[0, 0]+3000.0, accel[-1, 0]-3000.0, 1000.0/32)

    accel = np.stack([np.interp(timestamps, accel[:, 0], accel[:, 1]),
                      np.interp(timestamps, accel[:, 0], accel[:, 2]),
                      np.interp(timestamps, accel[:, 0], accel[:, 3])],
                     axis=1)

    bar_file = file_dict[(session[0], session[1], 'pressure.txt')][0]
    bar_df = pd.read_csv(dir_name + '/' + bar_file)
    bar = bar_df.drop_duplicates(bar_df.columns[0], keep='first').values
    bar = np.interp(timestamps, bar[:, 0], bar[:, 1]).reshape(-1, 1)

    # Apply lowess to smooth the barometer data with window-size 128
    # bar = np.convolve(bar[:, 0], np.ones(128)/128, mode='same').reshape(-1, 1)
    bar = sm.nonparametric.lowess(bar[:, 0], timestamps, return_sorted=False).reshape(-1, 1)

    # Keep data with dimension multiple of 128
    length_multiple_128 = 128*int(bar.shape[0]/128)
    accel = accel[0:length_multiple_128, :]
    bar = bar[0:length_multiple_128, :]
    labels = np.array(bar.shape[0]*[int(activity_indices[session[1]])]).reshape(-1, 1)
    timestamps = timestamps[0:length_multiple_128]

    if raw_data_features is None:
      raw_data_features = np.append(accel, bar, axis=1)
      raw_data_labels = labels
      interpolated_timestamps = timestamps
    else:
      raw_data_features = np.append(raw_data_features, np.append(accel, bar, axis=1), axis=0)
      raw_data_labels = np.append(raw_data_labels, labels, axis=0)
      interpolated_timestamps = np.append(interpolated_timestamps, timestamps, axis=0)

  return raw_data_features, raw_data_labels, interpolated_timestamps



def feature_extraction(raw_data_features, raw_data_labels, timestamps):
  """    raw_data_features: The fourth column is the barometer data.

  Returns:
    features: Features extracted from the data features, where
              features[:, 0] is the mean magnitude of acceleration;
              features[:, 1] is the variance of acceleration;
              features[:, 2:6] is the fft power spectrum of equally-spaced frequencies;
              features[: 6:12] is the fft power spectrum of frequencies in logarithmic sacle;
              features[:, 13] is the slope of pressure.
  Args:

  """
  features = None
  labels = None

  accel_magnitudes = np.sqrt((raw_data_features[:, 0]**2).reshape(-1, 1)+
                             (raw_data_features[:, 1]**2).reshape(-1, 1)+
                             (raw_data_features[:, 2]**2).reshape(-1, 1))

  # The window size for feature extraction
  segment_size = 128


  for i in range(0, accel_magnitudes.shape[0]-segment_size, 64):
    segment = accel_magnitudes[i:i+segment_size, :]
    accel_mean = np.mean(segment)
    accel_var = np.var(segment)

    segment_fft_powers = np.abs(np.fft.fft(segment))**2

    # Aggreate band power within frequency range, with equal space (window size=32) or logarithmic scale
    # Band power of equally-sapced bands
    equal_band_power = list()
    window_size = 32
    for j in range(0, len(segment_fft_powers), window_size):
      equal_band_power.append(sum(segment_fft_powers[j: j+32]).tolist()[0])

    # Band power of bands in logarithmic scale
    log_band_power = list()
    freqs = [0, 2, 4, 8, 16, 32, 64, 128]
    for j in range(len(freqs)-1):
      log_band_power.append(sum(segment_fft_powers[freqs[j]: freqs[j+1]]).tolist()[0])

    # Slope of barometer data
    # bar_slope = raw_data_features[i+segment_size-1, 3] - raw_data_features[i, 3]
    bar_slope = np.polyfit(timestamps[i:i+segment_size], raw_data_features[i:i+segment_size, 3], 1)[0]
    # bar_slope = np.polyfit([x*0.1 for x in range(segment_size)], raw_data_features[i:i+segment_size, 3], 1)[0]

    feature = [accel_mean, accel_var] + equal_band_power + log_band_power + [bar_slope]

    if features is None:
      features = np.array([feature])
    else:
      features = np.append(features, [feature], axis=0)

    label = Counter(raw_data_labels[i:i+segment_size][:, 0].tolist()).most_common(1)[0][0]

    if labels is None:
      labels = np.array([label])
    else:
      labels = np.append(labels, [label], axis=0)

  return features, labels




def five_fold_cross_validation(features, labels):

    true_labels = list()
    predicted_labels = list()
    clf = None  # Will store the last trained model

    for train_index, test_index in StratifiedKFold(n_splits=5).split(features, labels):
        X_train = features[train_index, :]
        Y_train = labels[train_index]

        X_test = features[test_index, :]
        Y_test = labels[test_index]

        clf = RandomForestClassifier(n_estimators=2, random_state=42)
        clf.fit(X_train, Y_train)
        predicted_label = clf.predict(X_test)

        predicted_labels += predicted_label.flatten().tolist()
        true_labels += Y_test.flatten().tolist()

    #final model and save it 
    
    final_clf = RandomForestClassifier(n_estimators=2, random_state=42)
    final_clf.fit(features, labels)
    
    #COPY THIS SECTION

    import joblib
    joblib.dump(final_clf, "fall_model.pkl")
    print("Sklearn model saved as fall_model.pkl")

    # Convert to Java code for Android 
    import m2cgen as m2c
    java_code = m2c.export_to_java(final_clf, class_name="FallDetectionModel")
    with open("FallDetectionModel.java", "w") as f:
        f.write(java_code)
    print("Java model saved as FallDetectionModel.java")



if __name__ == "__main__":

  data_path = './uploaded_falling_combined - Copy/'

  raw_data_features, raw_data_labels, timestamps = compute_raw_data(data_path)

  "***Feature Extraction***"
  features, labels = feature_extraction(raw_data_features, raw_data_labels, timestamps)

  
  five_fold_cross_validation(features, labels)
  
  
  

  

  






