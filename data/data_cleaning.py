import numpy as np
import pandas as pd

# %% define path constants
RAW_PATH = "./unprocessed/IMT_Classification_Dataset_matminer_and_handbuilt_v2.xlsx"
SAVE_PATH = "./processed/IMT_Classification_Dataset_Processed.xlsx"


# %% define helper functions
def remove_nullrows(df):
    """
    Remove rows with at least 6 missing columns

    :param df: Pandas DataFrame, the raw dataset
    :return: Pandas DataFrame, the processed dataset
    """
    # check if there is any observation without a class label
    if pd.isnull(df.Label).sum() != 0:
        raise Exception("Unlabelled compound(s) present!")

    return df.dropna(thresh=df.shape[1] - 6)  # drop rows with at least 6 columns missing


def mean_imputation(df):
    """Impute the missing numeric values with the mean of that column"""
    imputed_df = df.fillna(df.mean())
    return imputed_df


def abbreviate_features(df):
    """Remove the 'MagpieData ' prefix in the column names"""
    old_names = list(df)
    new_names = [name.replace("MagpieData ", "") for name in old_names]
    name_mapper = dict(zip(old_names, new_names))
    return df.rename(columns=name_mapper)


def remove_redundant_features(df):
    """Drop zero variance and highly correlated features from the dataset"""
    # drop the zero variance columns
    feature_variances = df.var()
    zero_var_columns = [column for column, variance in zip(feature_variances.index, feature_variances.values)
                        if variance == 0]
    df.drop(columns=zero_var_columns, inplace=True)

    # drop the highly correlated columns
    df_no_obj = df.drop(columns="Label").select_dtypes(exclude=object)  # drop the label and object/text columns
    # create the correlation matrix with absolute values
    corr_matrix = df_no_obj.corr().abs()
    # select only half of the correlation matrix
    half_corr_mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    half_corr_matrix = corr_matrix.mask(half_corr_mask)
    # find one column in each pair of columns with correlation > 0.95
    high_corr_columns = [feature for feature in half_corr_matrix.columns
                         if any(half_corr_matrix[feature] > 0.95)]
    return df.drop(columns=high_corr_columns)


# %% the data cleaning workflow
# read in the raw dataset
df_raw = pd.read_excel(RAW_PATH)
# remove rows with at least 6 null values
df_processed = remove_nullrows(df_raw)
# impute the missing values with the mean of the corresponding feature
df_processed = mean_imputation(df_processed)
# rename the columns for easier readability and plotting
df_processed = abbreviate_features(df_processed)
# remove columns with zero variance and high correlations
df_processed = remove_redundant_features(df_processed)
# save the processed dataset
df_processed.to_excel(SAVE_PATH, index=False)
