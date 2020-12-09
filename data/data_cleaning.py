import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


# %% define helper functions
def remove_nullrows(df, num_missing_cols=6):
    """
    Remove rows with at least a certain number of missing columns (default:6)

    :param df: Pandas DataFrame, the raw dataset
    :param num_missing_cols: Int, the number of missing columns qualified for dropping
    :return: Pandas DataFrame, the processed dataset
    """
    # check if there is any observation without a class label
    if pd.isnull(df.Label).sum() != 0:
        raise Exception("Unlabelled compound(s) present!")

    return df.dropna(thresh=df.shape[1] - num_missing_cols).reset_index(drop=True)  # drop qualified rows


def median_imputation(df):
    """Impute the missing numeric values with the median of that column after grouping by label"""
    imputed_df = df.copy()
    # get the numeric columns in the input df ("Label" and "struct_ordered" are excluded)
    numeric_cols = imputed_df.drop(columns=["Label", "struct_ordered"]).select_dtypes(include="number").columns
    # iterate over all the selected numeric columns and impute by median within each label group
    for numeric_col in numeric_cols:
        imputed_df[numeric_col] = imputed_df.groupby("Label")[numeric_col].apply(lambda x: x.fillna(x.median()))
    return imputed_df


def knn_imputation(df, num_neighbors=5, **kwargs):
    """Impute the missing numeric values with the n(default=5) nearest neighbors"""
    columns_to_drop = ["Compound", "Label", "struct_file_path"]
    # initialize the KNNImputer
    imputer = KNNImputer(n_neighbors=num_neighbors, **kwargs)
    # drop irrelevant columns that are not used to find the NN distances
    df_to_impute = df.drop(columns=columns_to_drop)
    # impute the dataframe
    df_imputed = imputer.fit_transform(df_to_impute)
    # add back the column names
    df_imputed = pd.DataFrame(df_imputed, columns=df_to_impute.columns)
    # add back the dropped columns
    df_combined = pd.concat([df[columns_to_drop], df_imputed], axis=1)
    return df_combined


def abbreviate_features(df):
    """Remove the 'MagpieData ' prefix in the column names and replace whitespace with underscore"""
    old_names = list(df)
    new_names = [name.replace("MagpieData ", "") for name in old_names]
    new_names = [name.replace(" ", "_") for name in new_names]
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
    # first drop the label and object/text columns
    df_no_obj = df.drop(columns=["Label", "struct_ordered", "struct_disordered"]).select_dtypes(exclude=object)
    # create the correlation matrix with absolute values
    corr_matrix = df_no_obj.corr().abs()
    # select only half of the correlation matrix
    half_corr_mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    half_corr_matrix = corr_matrix.mask(half_corr_mask)
    # find one column in each pair of columns with correlation > 0.95
    high_corr_columns = [feature for feature in half_corr_matrix.columns
                         if any(half_corr_matrix[feature] > 0.95)]
    return df.drop(columns=high_corr_columns)
