from typing import List

from airflow.decorators import task, dag
from airflow.models import Variable
from asimov.config import Config

dag_config = Config.model_validate_json(Variable.get("precp-config"))


@dag(tags=["asimov", "precipitation"])
def precipitation_model_development_dag():
    @task.kubernetes(
        name="entrypoint",
        **dag_config.task_config_airflow.model_dump(),
        env_vars=dag_config.task_env_vars,
        env_from=dag_config.task_env_from_secrets,
    )
    def entrypoint():
        import mlflow

        with mlflow.start_run(run_name="precipitation-model-dev"):
            run_handle = mlflow.active_run()
            return run_handle.info.run_id

    @task.kubernetes(
        name="preprocess",
        **dag_config.task_config_airflow.model_dump(),
        env_vars=dag_config.task_env_vars,
        env_from=dag_config.task_env_from_secrets,
    )
    def preprocess(mlflow_parent_run_id: str):
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split

        import sys

        sys.path.append("/opt/airflow")
        from helpers import get_feature_engineering_transformer, classifier

        import mlflow

        mlflow.autolog()

        with mlflow.start_run(
            run_name="preprocess", parent_run_id=mlflow_parent_run_id
        ):
            mlflow.artifacts.download_artifacts(
                artifact_uri="s3://datasets/precipitation/merged_final_df.csv",
                dst_path="/tmp/",
            )
            df = pd.read_csv("/tmp/merged_final_df.csv")
            df = df.rename(
                columns={
                    "Band 2_y": "Band 2",
                    "Band 3_y": "Band 3",
                    "Band 4_y": "Band 4",
                }
            )
            missing_values = (
                df[
                    [
                        "PRECIPITATION",
                        "Value IMERG",
                        "Value COT",
                        "Value CER",
                        "Value CWP",
                        "Value ASPECT",
                        "Value TPI",
                        "Value TWI",
                        "Band 1",
                        "Band 2",
                        "Band 3",
                        "Band 4",
                        "Value WVC",
                        "Value topoDiversity",
                        "Value landformsGreece",
                    ]
                ]
                .isna()
                .sum()
            )
            print(missing_values)

            df["Value IMERG"] = df["Value IMERG"] / 10
            df["month"] = pd.DatetimeIndex(df["Date"]).month
            df["Date"] = pd.to_datetime(df["Date"])

            df = df.dropna(
                subset=[
                    "PRECIPITATION",
                    "Value IMERG",
                    "Value COT",
                    "Value CER",
                    "Value CWP",
                    "Value ASPECT",
                    "Value TPI",
                    "Value TWI",
                    "Band 1",
                    "Band 2",
                    "Band 3",
                    "Band 4",
                    "Value WVC",
                    "Value topoDiversity",
                    "Value landformsGreece",
                ]
            )

            # Define Features and Target
            features = [
                "Value IMERG",
                "Value COT",
                "Value CER",
                "Value CWP",
                "Value ASPECT",
                "Value TPI",
                "Value TWI",
                "Band 1",
                "Band 2",
                "Band 3",
                "Band 4",
                "Value WVC",
                "Value topoDiversity",
                "Value landformsGreece",
                "month"
            ]
            df = df[features + ["PRECIPITATION"]]
            df["PRECIPITATION"] = np.log1p(
                df["PRECIPITATION"]
            )  # Log-transform target variable

            feature_engineering_transformer = get_feature_engineering_transformer()
            df = feature_engineering_transformer.transform(df)

            value_counts = df["PRECIPITATION"].value_counts()
            mlflow.log_dict(
                df["PRECIPITATION"].value_counts().to_dict(),
                artifact_file="value_counts.json",
            )

            classify_bin = classifier(value_counts)
            y_categories = df["PRECIPITATION"].map(classify_bin)

            print("Full dataset category distribution:")
            print(y_categories.value_counts())

            train_df, temp_df, categories_train, categories_temp = train_test_split(
                df,  # Combined X and y DataFrame
                y_categories,
                test_size=0.3,
                random_state=42,
                stratify=y_categories,
            )
            val_df, test_df, categories_val, categories_test = train_test_split(
                temp_df,
                categories_temp,
                test_size=0.5,
                random_state=41,
                stratify=categories_temp,
            )

            mlflow.log_table(train_df, artifact_file="precipitation_train.parquet")
            mlflow.log_table(temp_df, artifact_file="precipitation_temp.parquet")
            mlflow.log_table(val_df, artifact_file="precipitation_val.parquet")
            mlflow.log_table(test_df, artifact_file="precipitation_test.parquet")

            run_handle = mlflow.active_run()
            return run_handle.info.run_id

    @task.kubernetes(
        name="train",
        **dag_config.task_config_airflow.model_dump(),
        env_vars=dag_config.task_env_vars,
        env_from=dag_config.task_env_from_secrets,
    )
    def train(mlflow_parent_run_id: str, mlflow_run_id_deps: List[str]):
        import numpy as np
        import pandas as pd
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import QuantileTransformer
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.metrics import (
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )
        from scipy.stats import pearsonr
        from xgboost import XGBRegressor
        from scipy.stats import uniform, randint
        from denseweight import DenseWeight
        from imbalance_metrics import regression_metrics as rm

        import sys

        sys.path.append("/opt/airflow")
        from helpers import get_feature_engineering_transformer, classifier

        import mlflow

        with mlflow.start_run(
            run_name="train", parent_run_id=mlflow_parent_run_id
        ) as run:
            metrics_list = []

            alphas = np.linspace(0.0, 1.0, 10)
            best_model = None
            best_score = float("inf")

            feature_engineering_transformer = get_feature_engineering_transformer()
            pipeline = make_pipeline(
                feature_engineering_transformer,
                QuantileTransformer(n_quantiles=100, output_distribution="normal"),
                XGBRegressor(random_state=42, objective="reg:squarederror"),
            )

            train_df = mlflow.load_table(
                "precipitation_train.parquet", run_ids=mlflow_run_id_deps
            )
            test_df = mlflow.load_table(
                "precipitation_test.parquet", run_ids=mlflow_run_id_deps
            )

            X_train, y_train = (
                train_df.loc[:, train_df.columns != "PRECIPITATION"],
                train_df["PRECIPITATION"],
            )
            X_test, y_test = (
                test_df.loc[:, test_df.columns != "PRECIPITATION"],
                test_df["PRECIPITATION"],
            )

            vc_load = mlflow.artifacts.load_dict(
                f"runs:/{mlflow_run_id_deps[0]}/value_counts.json"
            )
            value_counts = {float(k): v for k, v in vc_load.items()}

            for alpha in alphas:
                print(f"\nTraining and evaluating with alpha = {alpha:.1f}")

                dw = DenseWeight(alpha=alpha)  # alpha must be tuned a = 0 no weight
                dense_weights = dw.fit(y_train.values)

                param_dist = {
                    "xgbregressor__n_estimators": randint(100, 300),
                    "xgbregressor__max_depth": randint(3, 7),
                    "xgbregressor__learning_rate": uniform(0.01, 0.2),
                    "xgbregressor__subsample": uniform(0.7, 0.3),
                    "xgbregressor__colsample_bytree": uniform(0.7, 0.3),
                }

                search = RandomizedSearchCV(
                    pipeline,
                    param_distributions=param_dist,
                    n_iter=20,
                    cv=10,
                    scoring="neg_root_mean_squared_error",
                    verbose=0,
                    n_jobs=-1,
                    random_state=42,
                )
                search.fit(X_train, y_train, xgbregressor__sample_weight=dense_weights)

                # Best model
                current_model = search.best_estimator_

                # Evaluate on Test Set
                y_test_pred_log = current_model.predict(X_test)
                y_test_pred = np.expm1(y_test_pred_log)
                y_test_actual = np.expm1(y_test)

                test_mae = mean_absolute_error(y_test_actual, y_test_pred)
                test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
                test_r2 = r2_score(y_test_actual, y_test_pred)
                test_corr, _ = pearsonr(y_test_actual, y_test_pred)
                test_sera = rm.sera(y_test_actual, y_test_pred)

                # Classify test set predictions into categories (using log-transformed values)
                classify_bin = classifier(value_counts)
                y_test_categories = y_test.map(classify_bin)

                # Calculate per-category metrics for the test set
                categories = ["other", "medium", "few", "many"]
                category_metrics = {}
                for category in categories:
                    idx = y_test_categories == category

                    # Ensure there are at least 2 samples for meaningful evaluation
                    if np.sum(idx) < 2:
                        print(f"Skipping '{category}' category (not enough samples).")
                        continue

                    y_actual_cat = y_test_actual[idx]
                    y_pred_cat = y_test_pred[idx]

                    mae_cat = mean_absolute_error(y_actual_cat, y_pred_cat)
                    rmse_cat = np.sqrt(mean_squared_error(y_actual_cat, y_pred_cat))
                    r2_cat = r2_score(y_actual_cat, y_pred_cat)
                    sera_cat = rm.sera(y_actual_cat, y_pred_cat)

                    category_metrics[f"{category}MAE"] = mae_cat
                    category_metrics[f"{category}RMSE"] = rmse_cat
                    category_metrics[f"{category}SERA"] = sera_cat

                # Store metrics in a dictionary
                metrics_dict = {
                    "alpha": alpha,
                    "testMAE": test_mae,
                    "testRMSE": test_rmse,
                    "testSERA": test_sera,
                    **category_metrics,
                }

                # Append metrics to the list
                metrics_list.append(metrics_dict)

                # Check if this model is the best based on test MAE
                if test_mae < best_score:
                    best_score = test_mae
                    best_model = current_model
                    best_alpha = alpha

            mlflow.sklearn.log_model(best_model, "best_XGBRegressor_model")

            print(
                f"\nBest model saved with alpha = {best_alpha:.1f} and test MAE = {best_score:.4f}"
            )

            metrics_df = pd.DataFrame(metrics_list)
            mlflow.log_table(
                metrics_df, artifact_file="model_metrics_XGB_logLabels.parquet"
            )

            metrics_df = metrics_df.loc[:, ~metrics_df.columns.str.startswith("other")]
            mlflow.log_table(metrics_df, artifact_file="model_metrics.parquet")

            # Filter out columns that start with "other" and the "alpha" column
            columns_to_normalize = [
                col
                for col in metrics_df.columns
                if not col.startswith("other") and col != "alpha"
            ]

            metrics_df_normalized = (
                metrics_df[columns_to_normalize]
                - metrics_df[columns_to_normalize].min()
            ) / (
                metrics_df[columns_to_normalize].max()
                - metrics_df[columns_to_normalize].min()
            )

            # Compute the composite score using normalized columns
            # For MAE, RMSE, and SERA, lower values are better, so we use negative weights
            metrics_df_normalized["composite_score"] = (
                -metrics_df_normalized.filter(like="MAE").sum(
                    axis=1
                )  # Negative for MAE
                + -metrics_df_normalized.filter(like="RMSE").sum(
                    axis=1
                )  # Negative for RMSE
                + -metrics_df_normalized.filter(like="SERA").sum(
                    axis=1
                )  # Negative for SERA
            )

            # Add the 'alpha' column back to the normalized DataFrame for reference
            metrics_df_normalized["alpha"] = metrics_df["alpha"]

            best_alpha_composite = metrics_df_normalized.loc[
                metrics_df_normalized["composite_score"].idxmax(), "alpha"
            ]

            print(f"\nBest alpha based on composite score: {best_alpha_composite:.1f}")
            mlflow.log_table(
                metrics_df_normalized, artifact_file="model_metrics_normalized.parquet"
            )

            handle = mlflow.active_run().info.run_id
            return handle

    master_run_id = entrypoint()
    preprocess_rid = preprocess(mlflow_parent_run_id=master_run_id)
    train_rid = train(
        mlflow_parent_run_id=master_run_id, mlflow_run_id_deps=[preprocess_rid]
    )


precipitation_model_development_dag()
