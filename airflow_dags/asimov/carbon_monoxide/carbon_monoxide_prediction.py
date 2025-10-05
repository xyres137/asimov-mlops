from airflow.decorators import task, dag
from airflow.models import Variable

from typing import List
from asimov.config import ModelDevelopmentConfig

dag_config = ModelDevelopmentConfig.model_validate_json(Variable.get("cm-config"))


@dag(tags=["asimov", "carbon-monoxide"])
def carbon_monoxide_model_development_dag(tune_enabled: bool, use_gpu: bool):
    @task.kubernetes(
        name="entrypoint",
        **dag_config.task_config_airflow.model_dump(),
        env_vars=dag_config.task_env_vars,
        env_from=dag_config.task_env_from_secrets,
    )
    def entrypoint():
        import mlflow

        with mlflow.start_run(run_name="carbon-monoxide-model-dev"):
            run_handle = mlflow.active_run()
            return run_handle.info.run_id

    @task.kubernetes(
        name="split",
        **dag_config.task_config_airflow.model_dump(),
        env_vars=dag_config.task_env_vars,
        env_from=dag_config.task_env_from_secrets,
    )
    def split(mlflow_parent_run_id: str):
        import mlflow

        mlflow.autolog()

        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split

        with mlflow.start_run(run_name="split", parent_run_id=mlflow_parent_run_id):
            mlflow.artifacts.download_artifacts(
                artifact_uri="s3://datasets/carbon-monoxide/CO_Dataset.csv",
                dst_path="/tmp/",
            )
            df = pd.read_csv("/tmp/CO_Dataset.csv")

            min_target = 0
            max_target = 100
            bin_interval = 0.1

            bins = np.arange(min_target, max_target + bin_interval, bin_interval)
            binned = pd.cut(df["Ground CO Concentration"], bins, include_lowest=True)
            binned = binned.astype(pd.CategoricalDtype(categories=binned.cat.categories.tolist() + ["Other"]))

            bin_frequencies = binned.value_counts()
            rare_bins = bin_frequencies[bin_frequencies < 2].index
            binned = binned.where(~binned.isin(rare_bins), other="Other")

            binned_classes = pd.DataFrame({"Value": df["Ground CO Concentration"], "Bin": binned})

            train_bin, test_bin = train_test_split(df, test_size=0.2, random_state=42, stratify=binned_classes[["Bin"]])
            mlflow.log_table(train_bin, artifact_file="train_dataset.parquet")
            mlflow.log_table(test_bin, artifact_file="test_dataset.parquet")

            handle = mlflow.active_run().info.run_id
            return handle

    @task.kubernetes(
        name="normalize",
        **dag_config.task_config_airflow.model_dump(),
        env_vars=dag_config.task_env_vars,
        env_from=dag_config.task_env_from_secrets,
    )
    def normalize(mlflow_parent_run_id: str, mlflow_run_id_deps: List[str]):
        import mlflow

        mlflow.autolog()

        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import StandardScaler

        with mlflow.start_run(run_name="normalize", parent_run_id=mlflow_parent_run_id):
            train_dataset = mlflow.load_table(artifact_file="train_dataset.parquet", run_ids=mlflow_run_id_deps)
            test_dataset = mlflow.load_table(artifact_file="test_dataset.parquet", run_ids=mlflow_run_id_deps)

            not_scale_cols = [
                "Day of Year",
                "Total Seconds",
                "Ground CO Concentration",
            ]  # Ground CO Concentration is the target
            scale_cols = train_dataset.drop(columns=not_scale_cols).columns

            for col, max_val in zip(["Day of Year", "Total Seconds"], [365, 86400]):
                train_dataset[col] = np.sin(2 * np.pi * train_dataset[col] / max_val)
                test_dataset[col] = np.sin(2 * np.pi * test_dataset[col] / max_val)

            scaler = StandardScaler()
            train_scaled = pd.DataFrame(scaler.fit_transform(train_dataset[scale_cols]), columns=scale_cols)
            test_scaled = pd.DataFrame(scaler.transform(test_dataset[scale_cols]), columns=scale_cols)

            train_normalized = pd.concat([train_scaled, train_dataset[not_scale_cols]], axis=1)
            test_normalized = pd.concat([test_scaled, test_dataset[not_scale_cols]], axis=1)

            mlflow.log_table(train_normalized, artifact_file="train_normalized.parquet")
            mlflow.log_table(test_normalized, artifact_file="test_normalized.parquet")

            handle = mlflow.active_run().info.run_id
            return handle

    @task.kubernetes(
        name="tune",
        **dag_config.task_config_airflow.model_dump(),
        env_vars=dag_config.task_env_vars,
        env_from=dag_config.task_env_from_secrets,
        container_resources={"limits": {"nvidia.com/gpu": "1"}} if use_gpu else None
    )
    def tune(mlflow_parent_run_id: str, mlflow_run_id_deps: List[str], enabled: bool):
        import random
        import pandas as pd
        import numpy as np
        from tqdm import tqdm
        from imbalance_metrics import regression_metrics as rm
        from smogn import phi, phi_ctrl_pts
        from deepforest import CascadeForestRegressor
        from denseweight import DenseWeight
        from sklearn.model_selection import ParameterGrid, train_test_split

        np.int = np.int32
        np.float = np.float64
        np.bool = np.bool_

        def generate_random_values(n: int, seed=None):
            if seed is not None:
                random.seed(seed)
            return [random.randint(0, 100) for _ in range(n)]

        def preprocess_target(y_train: pd.DataFrame) -> pd.DataFrame:
            phi_params_manual = phi_ctrl_pts(
                y=y_train,
            )

            y_phi = np.array(phi(y=y_train, ctrl_pts=phi_params_manual))

            y_class = np.where(
                y_phi == 0,
                "low",  # Assign 'low' if value < 0
                np.where(y_phi < 1, "medium", "high"),  # Assign 'medium' if 0 <= value < 1, else 'high'
            )

            return pd.DataFrame({"Value": y_train, "Class": y_class})

        def tuning_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> pd.DataFrame:
            relevance_classes = preprocess_target(y_train)
            param_grid = {"alpha": [x / 10 for x in range(11)]}  # alpha: 0.0 to 1.0
            param_combinations = list(ParameterGrid(param_grid))

            results = []

            num_folds = 3
            random_seeds = generate_random_values(num_folds, seed=42)

            for params in tqdm(param_combinations, desc="Hyperparameter Tuning"):
                fold_metrics = []
                dw = DenseWeight(alpha=params["alpha"])

                for seed in random_seeds:
                    (
                        X_train_fold,
                        X_test_fold,
                        y_train_fold,
                        y_test_fold,
                    ) = train_test_split(
                        X_train,
                        y_train,
                        test_size=0.2,
                        stratify=relevance_classes[["Class"]],
                        random_state=seed,
                    )

                    model = CascadeForestRegressor(
                        bin_subsample=300000,
                        max_layers=10,
                        n_trees=150,
                        n_estimators=2,
                        max_depth=50,
                        use_predictor=True,
                        predictor="xgboost",
                        predictor_kwargs={
                            "max_depth": 8,
                            "learning_rate": 0.05,
                            "colsample_bytree": 0.8,
                            "subsample": 0.8,
                            "n_estimators": 50,
                            "random_state": 42,
                            "verbosity": 0,
                        },
                        n_jobs=-1,
                        random_state=42,
                    )

                    weights = dw.fit(y_train_fold.to_numpy())
                    model.fit(
                        X_train_fold.select_dtypes(exclude=["object"]),
                        np.squeeze(y_train_fold),
                        sample_weight=weights,
                    )
                    y_pred = model.predict(X_test_fold)

                    sera = rm.sera(list(y_test_fold), np.squeeze(y_pred))
                    fold_metrics.append(sera)

                mean_metric = np.mean(fold_metrics)
                results.append((params, mean_metric))

            result_df = pd.DataFrame([(p["alpha"], v) for p, v in results], columns=["Alpha", "Mean SERA"])
            return result_df

        import mlflow

        mlflow.autolog()

        with mlflow.start_run(run_name="tune", parent_run_id=mlflow_parent_run_id):
            train_dataset = mlflow.load_table(artifact_file="train_normalized.parquet", run_ids=mlflow_run_id_deps)

            X_train = train_dataset.drop(columns=["Ground CO Concentration"])
            y_train = train_dataset["Ground CO Concentration"]

            import os

            if enabled:
                results_df = tuning_model(X_train, y_train)
            else:
                results_df = pd.DataFrame(
                    {
                        "Alpha": [
                            0.0,
                            0.1,
                            0.2,
                            0.3,
                            0.4,
                            0.5,
                            0.6,
                            0.7,
                            0.8,
                            0.9,
                            1.0,
                        ],
                        "Mean SERA": [
                            8894.8069089294,
                            8987.063773374784,
                            8997.074467348082,
                            8939.888736849047,
                            9315.1834516117,
                            9039.413774066872,
                            8566.440649333914,
                            8891.848524774265,
                            8658.79093023558,
                            8111.407318935996,
                            8870.747383252388,
                        ],
                    }
                )

            mlflow.log_table(results_df, artifact_file="alpha_k_fold_results.parquet")
            handle = mlflow.active_run().info.run_id
            return handle

    @task.kubernetes(
        name="train",
        **dag_config.task_config_airflow.model_dump(),
        env_vars=dag_config.task_env_vars,
        env_from=dag_config.task_env_from_secrets,
        container_resources={"limits": {"nvidia.com/gpu": "1"}} if use_gpu else None
    )
    def train(mlflow_parent_run_id: str, mlflow_run_id_deps: List[str]):
        import mlflow

        mlflow.autolog()

        from denseweight import DenseWeight
        from deepforest import CascadeForestRegressor
        import numpy as np

        np.int = np.int32
        np.float = np.float64
        np.bool = np.bool_

        def train_model(X_train, y_train, alpha):
            dw = DenseWeight(alpha=alpha)  # alpha must be tuned a = 0 no weight
            weights = dw.fit(y_train.to_numpy())

            model = CascadeForestRegressor(
                bin_subsample=300000,
                max_layers=10,
                n_trees=150,
                n_estimators=2,
                max_depth=50,
                use_predictor=True,
                predictor="xgboost",
                predictor_kwargs={
                    "max_depth": 8,
                    "learning_rate": 0.05,
                    "colsample_bytree": 0.8,
                    "subsample": 0.8,
                    "n_estimators": 50,
                    "random_state": 42,
                },
                n_jobs=-1,
                random_state=42,
            )

            model.fit(
                X_train.select_dtypes(exclude=["object"]),
                y_train.squeeze(),
                sample_weight=weights,
            )
            return model

        with mlflow.start_run(run_name="train", parent_run_id=mlflow_parent_run_id):
            train_dataset = mlflow.load_table(artifact_file="train_normalized.parquet", run_ids=mlflow_run_id_deps)
            X_train = train_dataset.drop(columns=["Ground CO Concentration"])
            y_train = train_dataset["Ground CO Concentration"]

            tuning = mlflow.load_table(artifact_file="alpha_k_fold_results.parquet", run_ids=mlflow_run_id_deps)
            min_row = tuning.loc[tuning["Mean SERA"].idxmin()]
            min_mean_sera, best_alpha = min_row["Mean SERA"], min_row["Alpha"]

            trained_model = train_model(X_train, y_train, alpha=best_alpha)
            trained_model.save("/tmp/trained_model")
            mlflow.log_artifacts("/tmp/trained_model", "carbon_monoxide_predictor_best_model_native")

            from mlflow.models import infer_signature

            signature = infer_signature(X_train, y_train)
            mlflow.sklearn.log_model(
                trained_model,
                artifact_path="carbon_monoxide_predictor_best_model",
                signature=signature,
            )

    master_run_id = entrypoint()

    split_rid = split(mlflow_parent_run_id=master_run_id)
    normalize_rid = normalize(mlflow_parent_run_id=master_run_id, mlflow_run_id_deps=[split_rid])
    tune_rid = tune(mlflow_parent_run_id=master_run_id, mlflow_run_id_deps=[normalize_rid], enabled=tune_enabled)
    train(mlflow_parent_run_id=master_run_id, mlflow_run_id_deps=[normalize_rid, tune_rid])


carbon_monoxide_model_development_dag(**dag_config.runtime_config.model_dump(exclude_none=True))
