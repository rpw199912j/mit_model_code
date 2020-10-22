import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.stats import iqr
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import auc, plot_roc_curve, plot_precision_recall_curve
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV

# # set up constants
# PROCESSED_PATH = "../data/processed/IMT_Classification_Dataset_Processed_v4.xlsx"
# TRAIN_RANDOM_SEED = 31415926
# SCORING_METRICS = ["precision_weighted", "recall_weighted", "roc_auc", "f1_weighted"]
# EVAL_RANDOM_SEEDS = np.arange(0, 10)
# NUM_FOLDS = 5
# SAVE_FIG_PATH = "../plots/"
#
# # read in the processed data
# df = pd.read_excel(PROCESSED_PATH)


def load_data(df_input, choice="Multiclass"):
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
    if choice in ["Metal", "Insulator", "MIT"]:
        try:
            overwrite_dict = binary_label_mapper[choice]
            df_output.replace({"Label": overwrite_dict}, inplace=True)
        except KeyError:
            raise Exception('Invalid choice name. Use "Metal", "Insulator" or "MIT')

    features = df_output.drop(columns=["Compound", "Label", "struct_file_path"])
    true_labels = df_output["Label"]

    return features, true_labels


# %% hyperparameter tuning with grid search
def tune_hyperparam(df_input, class_of_choice, seed, model=xgb.XGBClassifier, num_folds=5, param_grid=None,
                    scoring_metric_for_tuning="f1_weighted"):
    """
    Tune the hyperparameters for a binary classifier of a given class
    Return a dictionary with the best parameters

    :param df_input: Pandas DataFrame, the original input dataset
    :param class_of_choice: String, one of "Metal", "Insulator", "MIT"
    :param num_folds: Integer, the number of stratified folds (default: 5)
    :param seed: Integer, the random seed for reproducibility
    :param model: sklearn compatible model
    :param param_grid: Dictionary, the hyperparameter grid to search over
    :param scoring_metric_for_tuning: String or list of Strings, the tuning scoring metric
    :return: Dictionary, the best parameters
    """
    if class_of_choice == "Multiclass":
        print("\nTuning for {label} classifier".format(label=class_of_choice))
    else:
        print("\nTuning for {label} vs. non-{label} binary classifier".format(label=class_of_choice))

    X_features, y_labels = load_data(df_input, class_of_choice)
    if model.__name__ == "LogisticRegression":
        X_features = RobustScaler().fit_transform(X_features)

    # initialize weights to account for class imbalance in multiclass classification
    sample_weights = None

    # if no param_grid is specified, use the following default grid
    # for multiclass
    if (not param_grid) and (class_of_choice == "Multiclass"):
        param_grid = {
            "n_estimators": [10, 20, 30, 40, 80, 100, 150, 200],
            "max_depth": [2, 3, 4, 5],
            "learning_rate": np.logspace(-3, 2, num=6),
            # scale_pos_weight is not specified for xgboost multiclass classification
            "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "base_score": [0.3, 0.5, 0.7],
            "random_state": [seed]
        }
        # if the xgboost params is supplied, compute the weights for each sample in multiclass classification
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_labels)
    # for binary
    elif not param_grid:
        param_grid = {
            "n_estimators": [10, 20, 30, 40, 80, 100, 150, 200],
            "max_depth": [2, 3, 4, 5],
            "learning_rate": np.logspace(-3, 2, num=6),
            "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "scale_pos_weight": [np.sum(y_labels == 0) / np.sum(y_labels == 1)],
            "base_score": [0.3, 0.5, 0.7],
            "random_state": [seed]
        }
    elif model.__name__ == "GradientBoostingClassifier":
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_labels)

    # initialize the stratified k-folds
    grid_stratified_folds = StratifiedKFold(n_splits=num_folds, shuffle=True,
                                            random_state=seed)
    # initialize the xgboost classifier
    model_to_tune = model()
    # tune the model on 5 fold cv, with f1_weighted being the evaluation metric
    tune_grid = GridSearchCV(model_to_tune, param_grid=param_grid, n_jobs=-1,
                             scoring=scoring_metric_for_tuning, cv=grid_stratified_folds, verbose=1)
    tune_grid.fit(X_features, y_labels, sample_weight=sample_weights)

    return tune_grid.best_params_


# best_params = {choice: tune_hyperparam(df, choice, TRAIN_RANDOM_SEED)
#                for choice in ["Metal", "Insulator", "MIT"]}
# print(best_params)


# %% evaluate model performance with multiple metrics using the sklearn API
def evaluate_model_helper(df_input, choice, params, seed, scoring_metrics, eval_model, num_folds=10,
                          eval_method="robust", verbose=0, cv_generator=None):
    """
    Evaluate the model performance with the given parameters, cv and seed
    With the given metric list

    :param df_input: Pandas DataFrame, the original input dataset
    :param choice: String, one of "Metal", "Insulator", "MIT"
    :param params: Dictionary, the best parameters from hyperparameter tuning
    :param seed: Integer, the random seed for reproducibility
    :param scoring_metrics: List, a list of scoring metrics
    :param eval_model: sklearn model, the model to evaluate
    :param num_folds: Integer, the number of stratified folds (default: 10)
    :param eval_method: String, one of "robust", "standard"
    :param verbose: Int, if 1, print out the intermediate results
    :param cv_generator: Cross validator, if None will use stratified k-fold
    :return: Dictionary
    """

    X_features, y_labels = load_data(df_input, choice)
    if eval_model.__name__ == "LogisticRegression":
        X_features = RobustScaler().fit_transform(X_features)

    fit_params_dict = None
    # if (multiclass & XGBClassifier) | GradientBoostingClassifier, specify the sample weights
    if ((choice == "Multiclass") and (eval_model.__name__ == "XGBClassifier")) or \
            (eval_model.__name__ == "GradientBoostingClassifier"):
        fit_params_dict = {"sample_weight": compute_sample_weight(class_weight="balanced", y=y_labels)}

    # if cv_folds is not specified
    if not cv_generator:
        # initialize the stratified k-folds
        cv_generator = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

    # initialize the xgboost classifier with the tuned parameters
    model_to_eval = eval_model(**params[choice])
    # evaluate the tuned model with stratified k-fold cv
    cv_scores = cross_validate(model_to_eval, X_features, y_labels,
                               scoring=scoring_metrics, cv=cv_generator, error_score=np.nan,
                               fit_params=fit_params_dict)

    if verbose == 1:
        print("\nEvaluating the {label} vs. non-{label} binary classifier (seed={rand_seed})".format(
            label=choice, rand_seed=seed))
        if num_folds:
            print("For {} folds".format(num_folds))
        if eval_method == "robust":
            printout_lst = ["Median {}: {:0.2f} w/ IQR: {:0.2f}".format(
                metric, np.nanmedian(cv_scores["test_" + metric]), iqr(cv_scores["test_" + metric], nan_policy="omit")
            ) for metric in scoring_metrics]
        elif eval_method == "standard":
            printout_lst = ["Mean {}: {:0.2f} w/ std: {:0.2f}".format(
                metric, np.nanmean(cv_scores["test_" + metric]), np.nanstd(cv_scores["test_" + metric])
            ) for metric in scoring_metrics]

        print(*printout_lst, sep="\n")
        print("-----------------------------------\n")

    return {metric: cv_scores["test_" + metric] for metric in scoring_metrics}


# %%
def evaluate_model(random_seeds, metrics, class_of_choice, model=xgb.XGBClassifier, method="robust", **kwargs):
    """
    Evaluate the model performance with a list of seeds and metrics.

    :param random_seeds: List, list of random seeds
    :param metrics: List, list of evaluation metrics
    :param class_of_choice: String, one of "Metal", "Insulator", "MIT"
    :param model: sklearn model
    :param method: String, one of "robust", "standard"
    :param kwargs: addition parameters passed to eval_xgb_model_helper()
    :return: Pandas DataFrame, the evaluation results df alongwith the printout
    """
    print("\n----------------------------------------------------------------------")
    print("Model type: {}".format(model.__name__))
    if class_of_choice == "Multiclass":
        print("Evaluating the {label} classifier using the following seeds\n{seeds}".format(
            label=class_of_choice, seeds=random_seeds
        ))
        # if multiclass, change "roc_auc" to "roc_auc_ovr_weighted"
        metrics = [metric + "_ovr_weighted" if metric == "roc_auc" else metric for metric in metrics]
    else:
        print("Evaluating the {label} vs. non-{label} binary classifier using the following seeds\n{seeds}".format(
            label=class_of_choice, seeds=random_seeds
        ))

    all_metrics = [evaluate_model_helper(choice=class_of_choice, seed=seed, scoring_metrics=metrics, eval_model=model,
                                         eval_method=method, **kwargs)
                   for seed in random_seeds]
    print_lst = [return_metric_print(method, metric, all_metrics) for metric in metrics]
    print(*print_lst, sep="\n")
    return return_metric_df(model=model, metrics=metrics, pos_class=class_of_choice,
                            eval_method=method, all_metric_lst=all_metrics)


def return_metric_df_helper(eval_model, pos_class, eval_method, eval_metric, all_metric_lst):
    """
    The helper function for return_metric_df()

    :param eval_model: sklearn model
    :param pos_class: String, one of "Metal", "Insulator", "MIT"
    :param eval_method: String, one of "robust", "standard"
    :param eval_metric: String, a evaluation metric
    :param all_metric_lst: List of Dictionaries, all the eval_metrics of a given binary classifier
    :return: Tuple, (the name of the evaluation model, the positive class, the evaluation method,
                     the name of the evaluation metric, the value of the evaluation metric,
                     all the evaluation result of the given classifier)
    """
    complete_metric_lst = np.array([metric_seed[eval_metric] for metric_seed in all_metric_lst])
    if eval_method == "robust":
        metric_lst = np.median(complete_metric_lst, axis=1)
        return eval_model.__name__, pos_class, eval_metric, np.median(metric_lst), iqr(metric_lst), list(metric_lst)
    elif eval_method == "standard":
        metric_lst = np.mean(complete_metric_lst, axis=1)
        return eval_model.__name__, pos_class, eval_metric, np.mean(metric_lst), np.std(metric_lst), list(metric_lst)
    else:
        raise Exception('Invalid eval_method. Use "robust" or "standard"')


def return_metric_df(model, metrics, **kwargs):
    """
    Return the evaluation results in a dataframe

    :param model: sklearn model
    :param metrics: List, a list of evaluation metrics
    **kwargs: additional parameters to pass to return_metric_df_helper()
    :return: Pandas DataFrame
    """
    lst_metric_tuples = [return_metric_df_helper(eval_model=model, eval_metric=metric, **kwargs) for metric in metrics]
    return pd.DataFrame.from_records(lst_metric_tuples,
                                     columns=["model_type", "positive_class", "metric_name",
                                              "metric_value", "metric_dispersion", "raw_metric"])


def return_metric_print(eval_method, eval_metric, all_metric_lst):
    """
    Return the evaluation result of a evaluation metric

    :param eval_method: String, one of "robust", "standard"
    :param eval_metric: String, a evaluation metric
    :param all_metric_lst: List of Dictionaries, all the eval_metrics of a given binary classifier
    :return: String, the printout string of either of the two forms
        Median {eval_metric}: w/ IQR
        Mean {eval_metric} w/ std
    """
    if eval_method == "robust":
        metric_lst = np.median(np.array([metric_seed[eval_metric] for metric_seed in all_metric_lst]), axis=1)
        return "Median {}: {:0.2f} w/ IQR: {:0.2f}".format(eval_metric, np.median(metric_lst), iqr(metric_lst))
    elif eval_method == "standard":
        metric_lst = np.mean(np.array([metric_seed[eval_metric] for metric_seed in all_metric_lst]), axis=1)
        return "Mean {}: {:0.2f} w/ std: {:0.2f}".format(eval_metric, np.mean(metric_lst), np.std(metric_lst))
    else:
        raise Exception('Invalid eval_method. Use "robust" or "standard"')


# %%
# for choice in ["Metal", "Insulator", "MIT"]:
#     eval_model(EVAL_RANDOM_SEEDS, SCORING_METRICS, choice, df_input=df, params=best_params, method="robust",
#                num_folds=NUM_FOLDS)


# %% define a function to plot roc or precision_recall curves
def plot_eval(df_input, tuned_params, eval_seeds, num_folds=10, eval_method="roc", fontsize=24,
              individual_alpha=0.0, stat_func=np.median):
    """
    Plot the roc or precision_recall curves using stratified k-fold cv and a given list of random seeds

    :param df_input: Pandas DataFrame, the input dataframe
    :param tuned_params: Dictionary, the tuned hyperparameters
    :param eval_seeds: List, a list of random seeds
    :param num_folds: Integer, the number of stratified folds
    :param eval_method: String, "roc" or "pr"
    :param fontsize: Integer, the font size for the output plot
    :param individual_alpha: Float, the alpha (transparency) value for each individual fold curve
    :param stat_func: Function, np.median or np.mean
    :return: A Matplotlib figure
    """

    # initialize the stratified folds using the given list of seeds
    stratified_folds = [StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=eval_seed)
                        for eval_seed in eval_seeds]
    # intialize lists for colors
    color_lst = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # construct the plot
    fig, ax = plt.subplots(3, 1, sharex="all", sharey="all",
                           figsize=(10, 10))

    # iterate over the 3 binary classifiers
    for index, (choice, title) in enumerate(zip(["Metal", "Insulator", "MIT"],
                                                ["M", "I", "T"])):
        # initialize the eval_metric list
        aucs = []
        mean_x = np.linspace(0, 1, 100)
        # load in the data for the specified binary classifier
        X_choice, y_choice = load_data(df_input, choice)
        # initialize the tuned binary classifier
        tuned_xgb_model = xgb.XGBClassifier(**tuned_params[choice])
        # setup naive precision if eval_method == "pr"
        if eval_method == "pr":
            naive_precision = np.sum(y_choice == 1) / len(y_choice)
        # iterate over the stratified folds created using the random seeds in eval_seeds
        for stratified_fold in stratified_folds:
            aucs_seed = []
            y = []
            for train_indices, test_indices in stratified_fold.split(X_choice, y_choice):
                # fit the model on k-1 training folds
                tuned_xgb_model.fit(X_choice.iloc[train_indices], y_choice[train_indices])
                # plot the pr curve on the 1 test fold
                if eval_method == "roc":
                    plot_func = plot_roc_curve
                elif eval_method == "pr":
                    plot_func = plot_precision_recall_curve
                else:
                    raise Exception('Invalid eval_method. Please choose from "roc" or "pr"')
                # create the visulization object
                viz = plot_func(tuned_xgb_model,
                                X_choice.iloc[test_indices], y_choice[test_indices],
                                ax=ax[index], alpha=individual_alpha)
                if eval_method == "roc":
                    # get the AUC score for that test fold
                    aucs_seed.append(viz.roc_auc)
                    # linear interpolation for true positive rate
                    interp_y = np.interp(mean_x, viz.fpr, viz.tpr)
                    # set the first interpolated true positive rate to be 0
                    interp_y[0] = 0.0
                    # get the linearly interpolated true positive rate
                    y.append(interp_y)
                    # get the average/median roc curve of a given seed
                    mean_y = stat_func(y, axis=0)
                    mean_y[-1] = 1
                elif eval_method == "pr":
                    # get the AUC score for that test fold
                    aucs_seed.append(auc(viz.recall, viz.precision))
                    # linear interpolation for precision
                    interp_y = np.interp(mean_x, viz.recall[::-1], viz.precision[::-1])
                    # set the first interpolated precision to be 1
                    interp_y[0] = 1.0
                    # get the linearly interpolated precision
                    y.append(interp_y)
                    # get the average/median pr curve of a given seed
                    mean_y = stat_func(y, axis=0)
                    mean_y[-1] = 0

            ax[index].plot(mean_x, mean_y, alpha=0.6, lw=3, color=color_lst[index])
            aucs.append(stat_func(aucs_seed))

        if eval_method == "roc":
            ax[index].set(xlim=(-0.01, 1), ylim=(0, 1.05), xlabel=None, ylabel=None)
            ax[index].set_title(title, fontsize=fontsize)
            ax[index].plot([0, 1], [0, 1], linestyle="--", lw=3, color="k")
            ax[index].text(0.98, 0.05, "{} AUC: {:.2f}".format(stat_func.__name__.capitalize(),
                                                               stat_func(aucs)), fontsize=fontsize,
                           horizontalalignment="right", verticalalignment="bottom")
        elif eval_method == "pr":
            ax[index].set(xlim=(0, 1), ylim=(0, 1.05), xlabel=None, ylabel=None)
            ax[index].set_title("{} {} AUC: {:.2f}".format(title, stat_func.__name__.capitalize(),
                                                           stat_func(aucs)), fontsize=fontsize)
            ax[index].plot([0, 1], [naive_precision, naive_precision],
                           linestyle="--", lw=3, color="k")
            ax[index].text(0.02, naive_precision - 0.03, "naive precision:{:.2f}".format(naive_precision),
                           fontsize=fontsize - 4, horizontalalignment="left", verticalalignment="top")

        ax[index].get_legend().remove()
        ax[index].tick_params(axis='both', which='major', labelsize=fontsize)

    if eval_method == "roc":
        x_axis_label = "False Positive Rate"
        y_axis_label = "True Positive Rate"
    elif eval_method == "pr":
        x_axis_label = "Recall"
        y_axis_label = "Precision"
    ax[2].set_xlabel(x_axis_label, fontsize=fontsize)
    ax[1].set_ylabel(y_axis_label, fontsize=fontsize)
    plt.tight_layout()
    plt.show()

    return fig


# roc_curve = plot_eval(df, best_params, eval_seeds=EVAL_RANDOM_SEEDS, num_folds=NUM_FOLDS)
# roc_curve.savefig(SAVE_FIG_PATH + "roc_curve_10_seeds.pdf", dpi=300)
#
# # %%
# pr_curve = plot_eval(df, best_params, eval_seeds=EVAL_RANDOM_SEEDS, num_folds=NUM_FOLDS, eval_method="pr")
# pr_curve.savefig(SAVE_FIG_PATH + "pr_curve_10_seeds.pdf", dpi=300)
#
# # %%
# plot_eval(df, best_params, eval_seeds=[TRAIN_RANDOM_SEED], num_folds=NUM_FOLDS, stat_func=np.median,
#           individual_alpha=0.5)

