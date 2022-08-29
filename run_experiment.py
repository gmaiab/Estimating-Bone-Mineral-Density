import json
import os
import time
from datetime import datetime
from typing import Tuple, Union

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (explained_variance_score, max_error,
                             mean_absolute_error, mean_squared_error,
                             median_absolute_error, r2_score)
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models import MODELS, MODELS_PARAMS, RANDOM_STATE
from utils.logger import Logger


def run_experiments(
    name: str,
    data: pd.DataFrame, 
    scaler: Union[ColumnTransformer, StandardScaler], 
    metrics: Tuple, 
    scoring: str, 
    outer_k: int, 
    inner_k: int,
    base_output_folder: str = 'output'
    ) -> None:

    Logger().info("Creating a directory for saving info from the experiment")
    folder_name = f"{name}_{datetime.now()}"
    output_folder = os.path.join(base_output_folder, folder_name)
    os.makedirs(output_folder, exist_ok=True)
    Logger().info(f"Output info will be saved at {output_folder}")

    experiment_config = f"name: {name}\nscaler: {scaler}\nmetrics: {metrics}\nscoring: {scoring}\nouter_k and inner_k: {outer_k} and {inner_k}"
    experiment_config_path = os.path.join(output_folder, 'experiment_config.txt')
    with open(experiment_config_path, 'w') as f:
        f.write(experiment_config)
    Logger().info(f"Saved experiement config at {experiment_config_path}")

    # Split into features and labels
    X, y = data.iloc[:, :-1], data.iloc[:, [-1]]

    # Dictionary with the metrics being used and an empty list for each model
    Logger().info(f"Metrics to be used: {metrics}")
    df_results = pd.DataFrame()
    metrics_summary = {}
    for metric in metrics:
        metric_name = metric.__name__
        metrics_summary[metric_name] = {}

        for model_name in MODELS:
            metrics_summary[metric_name][model_name] = []

    list_feature_importance = []

    # Dicts with all the models pipelines and their search params
    Logger().info(f"Models to be tested: {MODELS.items()}")
    Logger().info(f"Models hyperparameters to be searched: {MODELS_PARAMS.items()}")
    models_pipeline = {}
    params_pipeline = {}
    for model_name, model in MODELS.items():
        pipeline = Pipeline(
            [
                ("scaler", scaler),
                (model_name, model)
            ]
        )
        models_pipeline[model_name] = pipeline
        
        search_params = {}
        for param_name, parameters in MODELS_PARAMS[model_name].items():
            pipeline_param_name = f"{model_name}__{param_name}"
            search_params[pipeline_param_name] = parameters
        params_pipeline[model_name] = search_params

    Logger().info(f"{outer_k} outer folds and {inner_k} inner folds")
    # Configuration of the cross-validation
    cv_outer = KFold(n_splits=outer_k, shuffle=True, random_state=RANDOM_STATE)
    cv_inner = KFold(n_splits=inner_k, shuffle=True, random_state=RANDOM_STATE)

    Logger().info("Experiment started")
    # Apply the experiment to all models
    for model_name, pipeline in models_pipeline.items():
        try:
            Logger().info(f"Creating output folder for model {model_name}")
            model_output_folder = os.path.join(output_folder, model_name)
            os.makedirs(model_output_folder, exist_ok=True)

            Logger().info(f"{model_name} - Outer CV started")
            nested_cv_start_time = time.time()
            for i, (train_ix, test_ix) in enumerate(cv_outer.split(X)):
                Logger().info(f"{model_name} - =========== SPLIT #{i+1} ===========")
                # split data
                X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
                y_train, y_test = y.iloc[train_ix].values.ravel(), y.iloc[test_ix].values.ravel()

                # define search
                search = GridSearchCV(
                    estimator=pipeline, 
                    param_grid=params_pipeline[model_name], 
                    scoring=scoring,
                    n_jobs=-1,
                    refit=True,
                    cv=cv_inner,
                    verbose=1
                )
                
                # execute search
                Logger().info(f"{model_name} - Inner CV fit started")
                search_start_time = time.time()
                result = search.fit(X_train, y_train)
                search_time = time.time() - search_start_time
                Logger().info(f"{model_name} - Inner CV fit finished")

                # get results summary from CV
                cv_results = result.cv_results_
                
                # get the best performing model fit on the whole training set
                best_model = result.best_estimator_
                refit_time = result.refit_time_
                best_score = result.best_score_
                best_params = result.best_params_
                Logger().info(f"{model_name} - Best score in the search: {best_score}. Best params: {best_params}")

                # evaluate model on the hold out dataset
                Logger().info(f"{model_name} - Inner CV prediction started")
                pred_start_time = time.time()
                y_pred = best_model.predict(X_test)
                pred_time = time.time() - pred_start_time
                Logger().info(f"{model_name} - Inner CV prediction finished")

                # calculate metrics
                Logger().info(f"{model_name} - Calculating the metrics")
                for metric in metrics:
                    result = metric(y_test, y_pred)
                    metric_name = metric.__name__
                    metrics_summary[metric_name][model_name].append(result)
                    Logger().info(f"{model_name} - {metric_name}: {result:.3f}")
                
                Logger().info(f"{model_name} - Times - Search time: {search_time:.3f}. Refit time: {refit_time:.4f}. Predict time: {pred_time}")
                
                # saving results
                Logger().info(f"{model_name} - Saving results")
                file_model = f"best_model_{i}.joblib"
                file_results = f"results_summary_fold_{i}.json"
                file_cv_results = f"cv_results_fold_{i}.csv"
                file_targets_pred = f"targets_predictions_{i}.csv"

                path_model = os.path.join(model_output_folder, file_model)
                path_results = os.path.join(model_output_folder, file_results)
                path_cv_results = os.path.join(model_output_folder, file_cv_results)
                path_targets_pred = os.path.join(model_output_folder, file_targets_pred)

                fold_results = {
                    'best_score': best_score,
                    'best_params': best_params,
                    'search_time': search_time,
                    'refit_time': refit_time,
                    'pred_time': pred_time,
                }

                with open(path_results, 'w') as f:
                    json.dump(fold_results, f)
                Logger().info(f"{model_name} - Results summary saved at {path_results}")

                df_cv_results = pd.DataFrame(cv_results)
                df_cv_results.to_csv(path_cv_results, encoding='utf-8', index=False)
                Logger().info(f"{model_name} - CV results saved at {path_cv_results}")

                targets = {
                    'indices': test_ix,
                    'y_test': y_test,
                    'y_pred': y_pred
                }
                df_targets = pd.DataFrame(targets)
                df_targets.to_csv(path_targets_pred, encoding='utf-8', index=False)
                Logger().info(f"{model_name} - Predictions saved at {path_targets_pred}")

                dump(best_model, path_model) 
                Logger().info(f"{model_name} - Best model saved at {path_model}")

            # compute the time for the nested cv
            nested_cv_time = time.time() - nested_cv_start_time
            Logger().info(f"{model_name} - Outer CV finished")
            Logger().info(f"{model_name} - Time - Nested CV took {nested_cv_time:.2f} seconds")

            # display metrics
            Logger().info(f"{model_name} - Computing final metrics")
            
            dict_results = {}
            dict_results['model_name'] = model_name
            for metric in metrics:
                metric_name = metric.__name__
                metric_values = metrics_summary[metric_name][model_name]
                metric_mean = np.mean(metric_values)
                metric_std = np.std(metric_values)
                dict_results[f"{metric_name}_mean"] = metric_mean
                dict_results[f"{metric_name}_std"] = metric_std
                dict_results[f"{metric_name}"] = metric_values
                Logger().info(f"{model_name} - {metric_name} = {metric_mean} +- {metric_std}")

            df_results = df_results.append(dict_results, ignore_index=True)

            file_final_metrics = "final_metrics.csv"
            path_final_metrics = os.path.join(output_folder, file_final_metrics)
            df_results.to_csv(path_final_metrics, encoding='utf-8', index=False)
            
            Logger().info(f"{model_name} - Final metrics saved at {path_final_metrics}")

        except Exception as e:
            Logger().error(f"An error ocurred while testing model {model_name}. Error: {e}")

    Logger().info("Experiment finished")


if __name__ == "__main__":
    # Read the CSV data
    dataset_name = "dataset"
    src_metadata = f'data/{dataset_name}.csv'
    data = pd.read_csv(src_metadata)

    # Metrics functions to be calculated in the outer CV
    metrics = (r2_score, mean_absolute_error, mean_squared_error, max_error, median_absolute_error, explained_variance_score)

    # Scoring function to be used in the Grid Search (inner CV)
    scoring = 'r2'

    # Number of folds for the outer CV and inner CV
    outer_k = 2
    inner_k = 2

    # Define name for the output 
    name = f"{dataset_name}_{scoring}_outer_{outer_k}_inner_{inner_k}_random_{RANDOM_STATE}"


    run_experiments(
        name = name,
        data = data, 
        scaler = StandardScaler(), 
        metrics = metrics, 
        scoring = scoring, 
        outer_k = outer_k, 
        inner_k = inner_k,
        base_output_folder = 'output'
        )
