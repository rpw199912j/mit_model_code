import copy
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.stats import iqr
from sklearn.metrics import auc, plot_roc_curve, plot_precision_recall_curve
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV

# set up constants
PROCESSED_PATH = "../data/processed/IMT_Classification_Dataset_Processed.xlsx"
RANDOM_SEED = 31415926

# read in the processed data
df = pd.read_excel(PROCESSED_PATH)


def load_data(df_input, choice=None):
    """
    Load in the original dataset and convert into binary labelling if the choice of class is specified.
    Return a tuple with features and ground-truth labels

    :param df_input: Pandas DataFrame, the original dataset
    :param choice: String, one of "Metal", "Insulator", "MIT"
    :return:
        features: Pandas DataFrame, the dataset containing all the features
        true_labels: Pandas Series, the class labels
    """

    df_output = df_input.copy()
    binary_label_mapper = {"Metal": {0: 1, 1: 0, 2: 0},
                           "Insulator": {2: 0},
                           "MIT": {1: 0, 2: 1}}
    if choice:
        try:
            overwrite_dict = binary_label_mapper[choice]
            df_output.replace({"Label": overwrite_dict}, inplace=True)
        except KeyError:
            raise Exception('Invalid choice name. Use "Metal", "Insulator" or "MIT')

    features = df_output.drop(columns=["Compound", "Label", "struct_file_path"])
    true_labels = df_output["Label"]

    return features, true_labels


# %% hyperparameter tuning with grid search
def tune_hyperparam(df_input, class_of_choice, seed, num_folds=5):
    """
    Tune the hyperparameters for a binary classifier of a given class
    Return a dictionary with the best parameters

    :param df_input: Pandas DataFrame, the original input dataset
    :param class_of_choice: String, one of "Metal", "Insulator", "MIT"
    :param num_folds: Integer, the number of stratified folds (default: 5)
    :param seed: Integer, the random seed for reproducibility
    :return: Dictionary, the best parameters
    """

    print("\nTuning for {label} vs. non-{label} binary classifier".format(label=class_of_choice))
    X_features, y_labels = load_data(df_input, class_of_choice)

    xgb_param_grid = {
        "n_estimators": [10, 20, 30, 40, 80, 100, 150, 200],
        "max_depth": [2, 3, 5],
        "learning_rate": np.logspace(-3, 2, num=6),
        "scale_pos_weight": [np.sum(y_labels == 0) / np.sum(y_labels == 1)],
        "base_score": [0.3, 0.5, 0.7]
    }

    # initialize the stratified k-folds  (default k=5)
    grid_stratified_folds = StratifiedKFold(n_splits=num_folds, shuffle=True,
                                            random_state=seed)
    # initialize the xgboost classifier
    xgb_tune_model = xgb.XGBClassifier()
    # tune the model on 5 fold cv, with f1_macro being the evaluation metric
    xgb_grid = GridSearchCV(xgb_tune_model, param_grid=xgb_param_grid, n_jobs=-1,
                            scoring="f1_macro", cv=grid_stratified_folds, verbose=1)
    xgb_grid.fit(X_features, y_labels)

    return xgb_grid.best_params_


best_params = {choice: tune_hyperparam(df, choice, RANDOM_SEED)
               for choice in ["Metal", "Insulator", "MIT"]}
print(best_params)


# %% evaluate model performance with multiple metrics using the sklearn API
def eval_xgb_model(df_input, class_of_choice, params, seed, num_folds=10, eval_method="robust"):
    """
    Evaluate the model performance with the given parameters using stratified k-fold cv.
    Return 4 metrics: test_precision_macro, test_recall_macro, test_roc_auc, test_f1_macro

    :param df_input: Pandas DataFrame, the original input dataset
    :param class_of_choice: String, one of "Metal", "Insulator", "MIT"
    :param params: Dictionary, the best parameters from hyperparameter tuning
    :param seed: Integer, the random seed for reproducibility
    :param num_folds: Integer, the number of stratified folds (default: 10)
    :param eval_method: String, one of "robust", "standard"
    :return: None, only the printout of metrics in either of the two forms
        "robust": median w/ IQR
        "standard": mean w/ std
    """

    print("\nEvaluating the {label} vs. non-{label} binary classifier".format(label=class_of_choice))
    print("For {} folds".format(num_folds))
    X_features, y_labels = load_data(df_input, class_of_choice)

    # define a list of evaluation metrics
    scoring_metrics = ["precision_macro", "recall_macro", "roc_auc", "f1_macro"]
    # initialize the stratified k-folds  (default k=10)
    stratified_folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    # initialize the xgboost classifier with the tuned parameters
    xgb_eval_model = xgb.XGBClassifier(**params[class_of_choice])
    # evaluate the tuned model with stratified k-fold cv
    cv_scores = cross_validate(xgb_eval_model, X_features, y_labels,
                               scoring=scoring_metrics, cv=stratified_folds)

    if eval_method == "robust":
        printout_lst = ["Median {}: {:0.2f} w/ IQR: {:0.2f}".format(
            metric, np.median(cv_scores["test_" + metric]), iqr(cv_scores["test_" + metric])
        ) for metric in scoring_metrics]
    elif eval_method == "standard":
        printout_lst = ["Mean {}: {:0.2f} w/ std: {:0.2f}".format(
            metric, np.mean(cv_scores["test_" + metric]), np.std(cv_scores["test_" + metric])
        ) for metric in scoring_metrics]
    else:
        raise Exception('Invalid eval_method. Use "robust" or "standard"')

    print(*printout_lst, sep="\n")


for choice in ["Metal", "Insulator", "MIT"]:
    eval_xgb_model(df, choice, best_params, RANDOM_SEED, eval_method="standard")

# %% Create the visulization for the ROC curves
# initialize the stratified folds
roc_stratified_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# intialize lists for colors
color_lst = ['#1f77b4', '#ff7f0e', '#2ca02c']

# construct the plot
fig, ax = plt.subplots(3, 1, sharex="all", sharey="all",
                       figsize=(10, 10))

for index, (choice, title) in enumerate(zip(["Metal", "Insulator", "MIT"],
                                            ["M", "I", "T"])):
    aucs = []
    X_choice, y_choice = load_data(df, choice)
    # initialize the tuned binary classifier
    roc_xgb_model = xgb.XGBClassifier(**best_params[choice])
    for i, (train_indices, test_indices) in enumerate(roc_stratified_folds.split(X_choice, y_choice)):
        # fit the model on k-1 training folds
        roc_xgb_model.fit(X_choice.iloc[train_indices], y_choice[train_indices])
        # plot the ROC curve on the 1 test fold
        viz = plot_roc_curve(roc_xgb_model, X_choice.iloc[test_indices], y_choice[test_indices],
                             alpha=0.6, lw=3, ax=ax[index], color=color_lst[index])
        # get the AUC score for that test fold
        aucs.append(viz.roc_auc)
    ax[index].set(xlim=(-0.01, 1), ylim=(0, 1.05), xlabel=None, ylabel=None)
    ax[index].set_title(title, fontsize=24)
    ax[index].get_legend().remove()
    ax[index].plot([0, 1], [0, 1], linestyle="--", lw=3, color="k")
    ax[index].tick_params(axis='both', which='major', labelsize=24)
    ax[index].text(0.98, 0.05, "Mean AUC:{:.2f}".format(np.mean(aucs)), fontsize=24,
                   horizontalalignment="right", verticalalignment="bottom")

# ax[2].set_xlabel("False Positive Rate", fontsize=24)
# ax[1].set_ylabel("True Positive Rate", fontsize=24)
# plt.tight_layout()
# plt.show()

# %% Create the visulization for the precision-recall curves
# initialize the stratified folds
pr_stratified_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# intialize lists for colors
color_lst = ['#1f77b4', '#ff7f0e', '#2ca02c']

# construct the plot
fig, ax = plt.subplots(3, 1, sharex="all", sharey="all",
                       figsize=(10, 10))

for index, (choice, title) in enumerate(zip(["Metal", "Insulator", "MIT"],
                                            ["M", "I", "T"])):
    aucs = []
    X_choice, y_choice = load_data(df, choice)
    naive_precision = np.sum(y_choice == 1) / len(y_choice)
    # initialize the tuned binary classifier
    pr_xgb_model = xgb.XGBClassifier(**best_params[choice])
    for i, (train_indices, test_indices) in enumerate(pr_stratified_folds.split(X_choice, y_choice)):
        # fit the model on k-1 training folds
        pr_xgb_model.fit(X_choice.iloc[train_indices], y_choice[train_indices])
        # plot the PR curve on the 1 test fold
        viz = plot_precision_recall_curve(pr_xgb_model,
                                          X_choice.iloc[test_indices], y_choice[test_indices],
                                          alpha=0.6, lw=3, ax=ax[index], color=color_lst[index])
        # get the AUC score for that test fold
        aucs.append(auc(viz.recall, viz.precision))
    ax[index].set(xlim=(0, 1), ylim=(0, 1.05), xlabel=None, ylabel=None)
    ax[index].set_title("{}: mean auc:{:.2f}".format(title, np.mean(aucs)), fontsize=24)
    ax[index].get_legend().remove()
    ax[index].plot([0, 1], [naive_precision, naive_precision],
                   linestyle="--", lw=3, color="k")
    ax[index].tick_params(axis='both', which='major', labelsize=24)
    ax[index].text(0.02, naive_precision - 0.03, "naive precision:{:.2f}".format(naive_precision),
                   fontsize=20, horizontalalignment="left", verticalalignment="top")

ax[2].set_xlabel("Recall", fontsize=24)
ax[1].set_ylabel("Precision", fontsize=24)
plt.tight_layout()
plt.show()

# TODO: Figure out a way to plot the averages of averages
# TODO: learn the UMAP dimensionaility reduction

# %% train and save the models
for choice in ["Metal", "Insulator", "MIT"]:
    X, y = load_data(df, choice)
    xgb_tuned_model = xgb.XGBClassifier(**best_params[choice])
    xgb_tuned_model.fit(X, y)
    xgb_tuned_model.save_model("../model/saved_models/{}.model".format(choice.lower()))

# %%
metal_model = xgb.XGBClassifier()
metal_model.load_model("../model/saved_models/metal.model")

insulator_model = xgb.XGBClassifier()
insulator_model.load_model("../model/saved_models/insulator.model")

mit_model = xgb.XGBClassifier()
mit_model.load_model("../model/saved_models/mit.model")
X_test = df.iloc[[0, 115, 116]].drop(columns=["Compound", "Label", "struct_file_path"])
X_test.at[0, "estct"] = None
print(metal_model.predict(X_test))
print(insulator_model.predict(X_test))
print(mit_model.predict(X_test))

