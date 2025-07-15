import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


# Step 1: Filter patients under a minimum age
def filter_age(df, min_age=18):
    """Remove patients younger than the specified minimum age (default: 18)."""
    return df[df['age'] >= min_age].copy()


# Step 2: Remove rows with too many missing values
def remove_rows_with_many_missing(df, max_missing=10):
    """Remove rows (patients) that have more than the allowed number of missing values."""
    return df[df.isnull().sum(axis=1) < max_missing].copy()


# Step 3: Clip outliers using percentile thresholds
def clip_outliers(df, lower_percentile=2, upper_percentile=98):
    """Clip values outside of the given percentiles to reduce outlier impact."""
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        p2 = np.percentile(df[col].dropna(), lower_percentile)
        p98 = np.percentile(df[col].dropna(), upper_percentile)
        df[col] = df[col].clip(p2, p98)
    return df


# Step 4: Feature Engineering

#Step 4A: Created a new categorical feature age_group
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


# Step 4B: Simplify ethnicity categories
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


# Step 5: Drop highly correlated features
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


# Step 6: Feature Standardization and Normalization
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
                df[col] = np.log1p(df[col])
                cols_right_skewed_auto.append(col)

    return df


# Step 7: Encode categorical features
def encode_categorical_features(df):
    """
    One-hot encode selected categorical columns.
    If reference_columns is provided, ensures output matches these columns.
    """
    le = LabelEncoder()
    if 'age_group' in df.columns:
        df['age_group_encoded'] = le.fit_transform(df['age_group'].astype(str))
        df.drop(columns=['age_group'], inplace=True)
        
    one_hot_cols = ['first_service', 'ethnicity_simplified']
    df = pd.get_dummies(df, columns=[col for col in one_hot_cols if col in df.columns], drop_first=False)
    return df


def drop_unnecessary_columns(df):
    """
    Drops identifier and redundant columns from the dataframe.
    Specifically removes: ['icustay_id', 'hadm_id', 'subject_id', 'gender', 'ethnicity']
    while keeping 'is_male' as the only gender indicator.
    """
    columns_to_drop = ['icustay_id', 'hadm_id', 'subject_id', 'gender', 'ethnicity', 'age', 'age_rounded',
        'race_white', 'race_black', 'race_hispanic', 'race_other']
    existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]

    # Drop them safely
    df = df.drop(columns=existing_cols_to_drop)

    return df


# Main pipeline function
def prepare_data(df):
    """Run the complete preprocessing pipeline on the input DataFrame."""
    # df = filter_age(df)
    # df = remove_rows_with_many_missing(df)
    df = clip_outliers(df)
    df = add_age_columns(df)
    df = simplify_ethnicity(df)
    df = transform_and_standardize(df)
    df = drop_highly_correlated_features(df)
    #df = encode_categorical_features(df)
    df = drop_unnecessary_columns(df)

    return df

# def build_pipeline():
#     def get_cols_after_prepare(df):
#         df_prepared = prepare_data(df.copy())
#         numeric_cols = df_prepared.select_dtypes(include='number').columns
#         binary_cols = [c for c in numeric_cols if set(df_prepared[c].dropna().unique()) <= {0, 1}]
#         continuous_cols = [c for c in numeric_cols if c not in binary_cols]
#         return continuous_cols, binary_cols

#     def make_pipe(df_sample):
#         conts, bins = get_cols_after_prepare(df_sample)

#         ct = ColumnTransformer([
#             ('impute_and_scale', Pipeline([
#                 ('imputer', IterativeImputer(random_state=0)),
#                 ('scale', StandardScaler())
#             ]), conts),
#             ('binary_passthrough', 'passthrough', bins)
#         ], remainder='passthrough', verbose_feature_names_out=False )

#         return Pipeline([
#             ('prepare', FunctionTransformer(prepare_data, validate=False)),
#             ('transform', ct)
#         ])

#     return make_pipe


def build_pipeline():
    def get_cols_after_prepare(df):
        df2 = prepare_data(df.copy())
        num_cols = df2.select_dtypes(include='number').columns
        bin_cols = [c for c in num_cols if set(df2[c].dropna().unique()) <= {0,1}]
        cont_cols = [c for c in num_cols if c not in bin_cols]
        ord_cols  = ['age_group'] if 'age_group' in df2.columns else []
        ohe_cols  = [c for c in ['first_service','ethnicity_simplified'] if c in df2.columns]
        return cont_cols, bin_cols, ord_cols, ohe_cols

    def make_pipe(df_sample):
        conts, bins, ords, ohes = get_cols_after_prepare(df_sample)
        ct = ColumnTransformer([
            ('num', Pipeline([
                ('impute', IterativeImputer(random_state=0)),
                ('scale', StandardScaler())
            ]), conts),
            ('bin', 'passthrough', bins),
            ('ord', OrdinalEncoder(), ords),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ohes)
        ], remainder='drop', verbose_feature_names_out=False)

        return Pipeline([
            ('prep', FunctionTransformer(prepare_data, validate=False)),
            ('trans', ct)
        ])

    return make_pipe


