# This script performs data preprocessing steps for the Cars93 dataset.

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# Load the dataset from GitHub
cars_url = "http://raw.githubusercontent.com/bneelkamal/MachineLearning/refs/heads/main/DataFiles/Cars93.csv"
cars_df = pd.read_csv(cars_url, na_values=["NA", "", "NaN"], keep_default_na=False)

# 1.2 Handle Missing Values
# Replace missing values in numerical columns with the median.
# Replace missing values in categorical columns with the mode.
def handling_missing_values(df):
    df['Cylinders'] = pd.to_numeric(df['Cylinders'], errors='coerce').astype('Int64')
    for col in df.select_dtypes(include=['float64', 'int64', 'Int64']).columns:
        df[col] = df[col].fillna(df[col].median())
    df['Cylinders'] = pd.to_numeric(df['Cylinders'], errors='coerce').astype('int64')
    return df

# 1.3 Reduce Noise
# Clip numerical features to fall between the 5th and 95th percentiles.
def reducing_noise(df):
    for col in df.select_dtypes(include=['float64', 'int64']):
        df[col] = df[col].clip(lower=df[col].quantile(0.05), upper=df[col].quantile(0.95)).round(2)
    return df

# 1.4 Encode Categorical Features
# Use Label Encoding for ordinal features and One-Hot Encoding for nominal features.
def encoding_categorical_features(df):
    for col in ['AirBags']:
        df[col] = LabelEncoder().fit_transform(df[col])
    df = pd.get_dummies(df, columns=['Model', 'Type', 'Manufacturer', 'DriveTrain', 'Origin', 'Man.trans.avail'])
    return df

# 1.5 Normalize/Scale Features
# Use MinMaxScaler to normalize numerical features.
def minmax_normalizing_features(df):
    num_features = df.select_dtypes(include=['float64', 'int64']).columns
    df[num_features] = MinMaxScaler().fit_transform(df[num_features])
    return df

# 1.6 Split Data
# Split the dataset into train, validation, and test sets in the ratio of 70:20:10.
def split_data_train_validation_test(df):
    train, temp = train_test_split(df, test_size=0.3, random_state=42)
    validation, test = train_test_split(temp, test_size=0.33, random_state=42)
    return train, validation, test

# Apply preprocessing steps
cars_df = handling_missing_values(cars_df)
cars_df = reducing_noise(cars_df)
cars_df = encoding_categorical_features(cars_df)
cars_df = minmax_normalizing_features(cars_df)
train_df, validation_df, test_df = split_data_train_validation_test(cars_df)

# Save the preprocessed data (optional)
# train_df.to_csv("train_data.csv", index=False)
# validation_df.to_csv("validation_data.csv", index=False)
# test_df.to_csv("test_data.csv", index=False)
