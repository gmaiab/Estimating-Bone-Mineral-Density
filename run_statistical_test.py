import json
import os
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import pandas as pd
from autorank import autorank, create_report, latex_table, plot_stats
from sklearn.metrics import (explained_variance_score, max_error,
                             mean_absolute_error, mean_squared_error,
                             median_absolute_error, r2_score)

from models import MODELS, ModelsEnum

# Defining acronyms for the models
acronyms = {
    ModelsEnum.LINEAR.value: 'LR',
    ModelsEnum.ELASTIC_NET.value: 'EN',
    ModelsEnum.DECISION_TREE.value: 'DT',
    ModelsEnum.MLP.value: 'MLP',
    ModelsEnum.KNN.value: 'kNN',
    ModelsEnum.SVM_POLY.value: 'SVM-Poly',
    ModelsEnum.SVM_SIGMOID.value: 'SVM-Sigmoid',
    ModelsEnum.LSSVM_LINEAR.value: 'LSSVM-Linear',
    ModelsEnum.LSSVM_RBF.value: 'LSSVM-RBF',
    ModelsEnum.RANDOM_FOREST.value: 'RF',
    ModelsEnum.XGBOOST.value: 'GBDT',
    }

if __name__ == "__main__":
    # Folder containing the output of "run_experiment.py"
    folder_name = "dataset_r2_outer_2_inner_2_random_0" # INSERT FOLDER NAME HERE
    folder = f'output/{folder_name}'
    reference_csv = 'final_metrics.csv'

    # Output folder for the statistical tests results
    statistical_tests_folder = 'statistical_tests'

    # Metrics to be evaluated
    metrics = (r2_score, mean_absolute_error, mean_squared_error, max_error, median_absolute_error, explained_variance_score)
    metrics_columns = [f.__name__ for f in metrics]
    model_name_column = 'model_name'

    # Read the csv with the final results
    csv_path = os.path.join(folder, reference_csv)
    if os.path.exists(csv_path):
        # Create output folder
        output_folder = os.path.join(folder, statistical_tests_folder)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        df = pd.read_csv(csv_path)
        models_names = df[model_name_column].values

        # Run statistical tests for all metrics
        for metric in metrics_columns:
            # Create output folder for a specific metric
            metric_output_folder = os.path.join(output_folder, metric)
            os.makedirs(metric_output_folder)

            # Get the metric values from the outer loops for each model
            metric_values_string = df[metric].values
            metric_values = [json.loads(s) for s in metric_values_string]

            data = pd.DataFrame()

            for model_name, metric_value in zip(models_names, metric_values):
                if model_name in MODELS:
                    data[acronyms[model_name]] = metric_value

            # Check if metrics are sorted descending or ascending
            if metric in [r2_score.__name__, explained_variance_score.__name__]:
                order = 'descending'
            else:
                order = 'ascending'

            # Run the statiscal test and save the results
            try:
                result = autorank(data, alpha=0.05, verbose=False, order=order, force_mode='nonparametric')
                with open(os.path.join(metric_output_folder, 'result.txt'), 'w') as f:
                    f.write(str(result))

                with open(os.path.join(metric_output_folder, 'report.txt'), 'w') as f:
                    with redirect_stdout(f):
                        create_report(result)

                with open(os.path.join(metric_output_folder, 'latex.txt'), 'w') as f:
                    with redirect_stdout(f):
                        latex_table(result)
                try:
                    plot_stats(result, width=8)
                    plt.savefig(os.path.join(metric_output_folder, 'stats.png'))
                except Exception as e:
                    print(e)
                    plot_stats(result, allow_insignificant=True)
                    plt.savefig(os.path.join(metric_output_folder, 'stats_misleading.png'))
            except Exception as e:
                print(f"Failed. Error: {e}")


