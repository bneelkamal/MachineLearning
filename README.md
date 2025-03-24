1. Data Preprocessing

You have been provided with a CSV file "Cars93.csv." The given dataset is related to cars and 
contains 26 columns. In the given dataset, “Price” is the target variable (i.e., the output). The 
marks distribution according to the tasks are as follows: 
1. Assign a type to each of the following features (a) Model, (b) Type, (c) Max. Price and 
(d) Airbags from the following: ordinal/nominal/ratio/interval scale. 
2. Write a function to handle the missing values in the dataset (e.g., any NA, NaN values). 
3. Write a function to reduce noise (any error in the feature) in individual attributes. 
4. Write a function to encode all the categorical features in the dataset according to the 
type of variable jointly. 
5. Write a function to normalize / scale the features either individually or jointly. 
6. Write a function to create a random split of the data into train, validation and test sets in 
the ratio of [70:20:10].

Report

This report details the preprocessing steps applied to the "Cars93.csv" dataset. The dataset contains information on various car models and their attributes, with "Price" being the target variable.
1.1 Feature Type Identification
Feature	Scale Type	Reasoning
Model	Nominal	Represents car model names, which are categorical and have no inherent order.
Type	Nominal	Represents car types (e.g., Small, Midsize), which are categorical and lack order.
Max.Price	Ratio	Represents the maximum price, a continuous numerical variable with a true zero point.
AirBags	Ordinal	Represents airbag categories (None, Driver only, Driver & Passenger), which have a meaningful order.
	
1.2 Handling Missing Values
Identify Missing Values: Columns with missing values (NA/NaN) were identified using cars_df.isna().any().
Data Type Conversion: The 'Cylinders' column, initially imported as an object, was converted to numeric (Int64) using pd.to_numeric.
Numerical Features: Missing values in numerical features were filled with the median using cars_df[col].fillna(cars_df[col].median()). The median is robust to outliers and provides a central tendency measure.
Categorical Features: The original data has the 'Airbags' column with "none" values represented as NA during data import. Those were replaced to none based on the original data.

1.3 Noise Reduction
Outlier Handling: The reduce_noise function clips numerical feature values to fall between the 5th and 95th percentiles using df[col].clip(). This removes outliers that could skew the data.

1.4 Categorical Feature Encoding
Ordinal Features: Label encoding was applied to the "AirBags" feature using LabelEncoder(). This preserves the ordinal relationship between the categories.
Nominal Features: One-hot encoding was applied to nominal features ('Model', 'Type', 'Manufacturer', 'DriveTrain', 'Origin', 'Man.trans.avail') using pd.get_dummies(). This avoids creating artificial ordinal relationships.

1.5 Feature Normalization/Scaling
MinMax Scaling: The minmax_normalizing_features function applies Min-Max scaling to numerical features using MinMaxScaler(). This scales the features to a range of 0 to 1.

1.6 Data Splitting
Train-Validation-Test Split: The split_data_train_validation_test function splits the data into train, validation, and test sets in the ratio of 70:20:10 using train_test_split.
Splitting the data allows for model training on a validation set, and unbiased evaluation on a held-out test set. This ensures the model's ability to generalize to unseen data.

These preprocessing steps ensure that the Cars93 dataset is prepared for effective model training and evaluation. By addressing missing values, noise, categorical features, and scaling, the dataset is now in a format suitable for applying machine learning algorithms.
