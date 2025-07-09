import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

# Step 1: Filter patients under a minimum age
def filter_age(df, min_age=18):
    """Remove patients younger than the specified minimum age (default: 18)."""
    return df[df['age'] >= min_age].copy()


# Step 2: Remove rows with too many missing values
def remove_rows_with_many_missing(df, max_missing=10):
    """Remove rows (patients) that have more than the allowed number of missing values."""
    return df[df.isnull().sum(axis=1) < max_missing].copy()


# Step 3: Fill missing values using MICE imputation
def mice_imputation(df):
    """Use Multiple Imputation by Chained Equations (MICE) to fill in missing values for numeric features."""
    numeric_cols = df.select_dtypes(include='number').columns
    imputer = IterativeImputer(random_state=0)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df


# Step 4: Clip outliers using percentile thresholds
def clip_outliers(df, lower_percentile=2, upper_percentile=98):
    """Clip values outside of the given percentiles to reduce outlier impact."""
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        p2 = np.percentile(df[col].dropna(), lower_percentile)
        p98 = np.percentile(df[col].dropna(), upper_percentile)
        df[col] = df[col].clip(p2, p98)
    return df


# Step 5: Feature Engineering

#Step 5A: Created a new categorical feature age_group
def add_age_columns(df):
    """
    Creates 'age_rounded' per unique patient (subject_id),
    and assigns an 'age_group' category for each row in the dataset.
    """
    # Create unique patients dataframe with rounded age
    unique_patients = df.groupby('subject_id')['age'].first().reset_index()
    unique_patients['age_rounded'] = unique_patients['age'].round(0).astype(int)

    # Merge back to the original dataframe to have 'age_rounded' column
    df = df.merge(unique_patients[['subject_id', 'age_rounded']], on='subject_id', how='left')

    # Create age_group column based on age
    df['age_group'] = pd.cut(
        df['age'],
        bins=[20, 30, 40, 50, 60, 70, 80, 90, 100],
        labels=['21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91+'])
    return df


# Step 5B: Simplify ethnicity categories
def simplify_ethnicity(df):
    """Map detailed ethnicity values to simplified categories (White, Black, Hispanic, Asian, Unknown, Other)."""
    def map_ethnicity(value):
        if 'WHITE' in value:
            return 'White'
        elif 'BLACK' in value:
            return 'Black'
        elif 'HISPANIC' in value:
            return 'Hispanic'
        elif 'ASIAN' in value:
            return 'Asian'
        elif 'UNKNOWN' in value or 'UNABLE' in value:
            return 'Unknown'
        else:
            return 'Other'
    df['ethnicity_simplified'] = df['ethnicity'].apply(map_ethnicity)
    return df


# Step 6: Drop highly correlated features
def drop_highly_correlated_features(df):
    """Remove features with high correlation to reduce multicollinearity and redundancy."""
    columns_to_drop = [
        'glucose_max1', 'bun_max', 'bun_min', 'wbc_mean', 'wbc_min',
        'lactate_mean', 'lactate_min', 'platelet_max',
        'hematocrit_max', 'hematocrit_min', 'hemoglobin_min',
        'diasbp_min', 'diasbp_max', 'diasbp_mean',
        'sysbp_min', 'sysbp_max', 'sysbp_mean',
        'meanbp_min', 'meanbp_max',
        'tempc_min', 'tempc_max',
        'spo2_min', 'spo2_max',
        'heartrate_min', 'heartrate_max',
        'resprate_min', 'resprate_max',
        'bicarbonate_min', 'bicarbonate_max',
        'chloride_min', 'sodium_min', 'creatinine_min',
        'glucose_min', 'inr_min'
    ]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    return df


# Step 7: Feature Standardization and Normalization
def transform_and_standardize(df):
    """
    Detects right-skewed numeric features, applies log1p transformation to them,
    then standardizes the remaining numeric features (z-score scaling).
    Returns the transformed dataframe.
    """
    # Columns to exclude from transformations
    columns_distribution = [
        'icustay_id', 'ethnicity', 'ethnicity_simplified', 'is_male',
        'hadm_id', 'subject_id', 'thirtyday_expire_flag', 'icu_los', 'gender',
        'race_white', 'race_black', 'race_hispanic', 'race_other',
        'metastatic_cancer', 'diabetes', 'first_service', 'vent', 'age_group'
    ]

    cols_right_skewed_auto = []

    # Detect and transform right-skewed features
    for col in df.select_dtypes(include='number').columns:
        if col not in columns_distribution:
            skew_val = df[col].skew()
            if skew_val > 1:
                print(f"{col} is highly right skewed (skew={skew_val:.2f}), applying log transform.")
                df[col] = np.log1p(df[col])
                cols_right_skewed_auto.append(col)

    print("Columns log-transformed:", cols_right_skewed_auto)

    # Create list of numeric columns to standardize
    columns_to_standardize = [
        col for col in df.select_dtypes(include='number').columns
        if col not in columns_distribution
    ]

    # Apply z-score standardization to ALL numeric columns (including those already log-transformed)
    scaler = StandardScaler()
    df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])

    return df


# Step 8: Encode categorical features
def encode_categorical_features(df):
    """
    One-hot encode selected categorical columns.
    Keeps all categories (drop_first=False) which is preferred
    for clustering and tree-based models.
    """
    one_hot_cols = ['first_service', 'ethnicity_simplified']
    df = pd.get_dummies(df, columns=[col for col in one_hot_cols if col in df.columns], drop_first=False)
    return df


# Main pipeline function
def prepare_data(df):
    """Run the complete preprocessing pipeline on the input DataFrame."""
    df = filter_age(df)
    df = remove_rows_with_many_missing(df)
    df = mice_imputation(df)
    df = clip_outliers(df)
    df = add_age_columns(df)
    df = simplify_ethnicity(df)
    df = transform_and_standardize(df)
    df = drop_highly_correlated_features(df)
    df = encode_categorical_features(df)
    return df
