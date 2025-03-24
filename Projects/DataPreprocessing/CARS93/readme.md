# 🚗 Cars93 Data Preprocessing Project

Welcome to the **Cars93 Data Preprocessing Project**! This repository takes you on a journey through the preprocessing of the `Cars93.csv` dataset—a treasure trove of car-related data with 26 columns, where **"Price"** reigns supreme as our target variable. Buckle up as we clean, transform, and split this dataset into a machine-learning-ready masterpiece! 🚀

*Hop in, and let’s preprocess some data!*

---

## 🎯 Project Overview

The goal? Transform the raw `Cars93.csv` dataset into a polished gem 💎 ready for predictive modeling. Here’s what we’ve accomplished:

1. **Feature Typing**: Classified features into nominal, ordinal, ratio, or interval scales.  
2. **Missing Values**: Tackled pesky NA/NaN values with finesse.  
3. **Noise Reduction**: Smoothed out outliers for cleaner data.  
4. **Encoding**: Converted categorical features into numeric bliss.  
5. **Scaling**: Normalized features to keep everything in check.  
6. **Splitting**: Divided the data into train, validation, and test sets (70:20:10).  

---

## 🛠️ Preprocessing Steps

### 1. Feature Type Identification
We’ve identified the scale types of key features. Here’s the breakdown:

| Feature     | Scale Type | Why?                                                                 |
|-------------|------------|----------------------------------------------------------------------|
| **Model**   | Nominal    | Car model names—categorical with no order.                          |
| **Type**    | Nominal    | Car types (e.g., Small, Midsize)—no inherent ranking.               |
| **Max.Price**| Ratio      | Continuous numbers with a true zero—perfect for ratios!             |
| **AirBags** | Ordinal    | Categories (None, Driver only, Driver & Passenger) have an order.   |

> **Visual Insight**: Imagine a pie chart showcasing the distribution of car types! 🍰

---

### 2. Handling Missing Values
No NA/NaN can hide from us! Here’s how we cleaned up:
- **Detection**: Used `cars_df.isna().any()` to spot missing values.
- **Conversion**: Transformed 'Cylinders' from object to numeric (`Int64`).
- **Numerical Fix**: Filled missing numbers with the **median**—robust and outlier-proof.
- **Categorical Fix**: Replaced NA in 'AirBags' with "none" based on the original data.

*Say goodbye to those gaps!*

---

### 3. Noise Reduction
Outliers? Not on our watch!  
- **Method**: Clipped numerical values between the **5th and 95th percentiles** using `df[col].clip()`.  
- **Result**: Smoother data, fewer surprises.

> **Visual Idea**: A before-and-after box plot of a noisy feature like 'Max.Price' would shine here! 📊

---

### 4. Categorical Feature Encoding
Categorical data got a numeric makeover:
- **Ordinal**: `AirBags` was label-encoded to preserve its order (e.g., None=0, Driver only=1).
- **Nominal**: Features like 'Model', 'Type', and 'Manufacturer' were one-hot encoded with `pd.get_dummies()`—no fake hierarchies here!

*From strings to numbers, seamlessly!*

---

### 5. Feature Normalization
We scaled numerical features to a cozy `[0, 1]` range:
- **Tool**: `MinMaxScaler()` from scikit-learn.
- **Why**: Keeps all features on the same playing field for modeling.

> **Visual Tip**: A line graph showing feature ranges before and after scaling would pop! 📈

---

### 6. Data Splitting
Ready to train? We split the data like pros:
- **Ratio**: 70% train, 20% validation, 10% test.
- **Method**: Used `train_test_split` for a random, unbiased split.
- **Purpose**: Train on one set, tweak on validation, and test on unseen data.

*Perfectly portioned for success!*

---

## 📊 Results
The `Cars93` dataset is now a lean, mean, machine-learning machine! With missing values filled, noise reduced, categories encoded, and features scaled, it’s primed for action. Whether you're predicting car prices or exploring trends, this dataset is ready to roll. 🚘

---

## 🧰 Tools & Libraries
- **Python**: The engine behind it all.
- **Pandas**: For data wrangling.
- **NumPy**: For number crunching.
- **Scikit-learn**: For encoding, scaling, and splitting.
