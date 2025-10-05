from typing import List

from airflow.decorators import task, dag
from airflow.models import Variable

from asimov.config import ModelDevelopmentConfig

dag_config = ModelDevelopmentConfig.model_validate_json(Variable.get("af-config"))


@dag(tags=["asimov", "active-fires"])
def active_fires_model_development_dag(tune_enabled: bool, train_enabled: bool, use_gpu: bool):
    @task.kubernetes(
        name="entrypoint",
        **dag_config.task_config_airflow.model_dump(),
        env_vars=dag_config.task_env_vars,
        env_from=dag_config.task_env_from_secrets,
    )
    def entrypoint():
        import mlflow

        with mlflow.start_run(run_name="active-fires-model-dev"):
            run_handle = mlflow.active_run()
            return run_handle.info.run_id

    @task.kubernetes(
        name="tune",
        **dag_config.task_config_airflow.model_dump(),
        env_vars=dag_config.task_env_vars,
        env_from=dag_config.task_env_from_secrets,
        container_resources={"limits": {"nvidia.com/gpu": "1"}} if use_gpu else None
    )
    def tune(mlflow_parent_run_id: str, enabled: bool):
        import mlflow

        mlflow.autolog()

        import os
        import numpy as np
        import pandas as pd

        import tensorflow as tf

        tf.random.set_seed(42)

        from sklearn.model_selection import train_test_split
        from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

        import sys

        sys.path.append("/opt/airflow")
        from helpers import (
            load_samples,
            autoencoder,
            custom_binary_crossentropy,
            dice_loss,
            generate_random_values,
            data_generator,
            f1_m,
        )

        def autoencoder_objective(param_space):
            def combined_loss(y_true, y_pred):
                return (1 - param_space["alpha"]) * dice_loss(y_true, y_pred) + param_space[
                    "alpha"
                ] * custom_binary_crossentropy(y_true, y_pred)

            train_csv = load_samples(
                os.path.join(param_space["relative_path"], "train_dataset.csv"),
                os.path.join(param_space["relative_path"], "Tilled_Data"),
            )

            input_parameters = [i for i, j in space.items() if j == 1 and i in input_features]
            input_shape = (128, 128, len(input_parameters))

            model = autoencoder(
                input_shape=input_shape,
                filters=64,
                activation="relu",
                final_activation="sigmoid",
            )
            model.compile(
                optimizer=tf.keras.optimizers.Adam(param_space["learning_rate"]),
                loss=combined_loss,
                metrics=["accuracy", "binary_crossentropy", dice_loss],
            )

            train_df = pd.DataFrame(
                train_csv,
                columns=[
                    "MODIS_RAD",
                    "MODIS_FIRE",
                    "SEVIRI",
                    "SEVIRI_HRV",
                    "ELEVATION",
                    "SLOPE",
                    "ASPECT",
                    "TPI",
                    "FUEL_MAP",
                    "CANOPY_HEIGHT",
                    "CANOPY_COVER",
                    "Class_2",
                ],
            )

            batch_size = 16
            epochs = 5  # Fixed for optimization

            k = 2  # Number of folds
            seed = 42
            random_values = generate_random_values(k, seed)
            fold_metrics = []

            for i in range(k):
                training, cross_val = train_test_split(
                    train_csv,
                    test_size=0.2,
                    stratify=train_df["Class_2"],
                    random_state=random_values[i],
                )
                train_dataget = data_generator(training, batch_size=batch_size, input_features=input_parameters)
                cross_val_dataget = data_generator(cross_val, batch_size=batch_size, input_features=input_parameters)
                history = model.fit(
                    train_dataget,
                    steps_per_epoch=int(len(training) / batch_size),
                    validation_data=cross_val_dataget,
                    validation_steps=int(len(cross_val) / batch_size),
                    epochs=epochs,
                    shuffle=True,
                    verbose=0,
                )

                # Evaluate the model on the validation set
                score = model.evaluate(cross_val_dataget, verbose=0, steps=1)
                fold_metrics.append(score[3])  # Use Dice score

            # Log parameters and results
            print("Features:", param_space)

            # Objective: minimize the validation loss
            validation_loss = np.mean(fold_metrics)
            print("Validation Loss:", validation_loss)

            del train_dataget, cross_val_dataget
            tf.keras.backend.clear_session()

            return {"loss": validation_loss, "status": STATUS_OK}

        with mlflow.start_run(run_name="tune", parent_run_id=mlflow_parent_run_id):
            if enabled:
                mlflow.artifacts.download_artifacts(
                    artifact_uri="s3://datasets/active-fires",
                    dst_path="/tmp/",
                )
                input_features = ["seviri_ir39_"]
                space = {
                    "input_features": input_features,
                    "learning_rate": hp.loguniform("learning_rate", np.log(1e-4), np.log(1e-2)),
                    "alpha": hp.uniform("alpha", 0, 1),
                    "seviri_ir39_": 1,
                    "relative_path": "/tmp/active-fires",
                }
                # Ensure 'seviri_ir39_' is always included in input features
                for col in input_features:
                    if col != "seviri_ir39_":
                        space[col] = hp.choice(col, [0, 1])
                trials = Trials()
                best = fmin(
                    fn=autoencoder_objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=50,
                    trials=trials,
                )

            else:
                best = {"alpha": 0.816, "learning_rate": 0.00007}

            mlflow.log_dict(best, artifact_file="best_hyperparams.json")
            handle = mlflow.active_run().info.run_id
            return handle

    @task.kubernetes(
        name="train",
        **dag_config.task_config_airflow.model_dump(),
        env_vars=dag_config.task_env_vars,
        env_from=dag_config.task_env_from_secrets,
        container_resources={"limits": {"nvidia.com/gpu": "1"}} if use_gpu else None
    )
    def train(mlflow_parent_run_id: str, mlflow_run_id_deps: List[str], enabled: bool):
        import mlflow

        mlflow.autolog()

        import os
        import json

        import tensorflow as tf

        tf.random.set_seed(42)

        from tensorflow.keras.callbacks import (
            ModelCheckpoint,
            ReduceLROnPlateau,
            EarlyStopping,
        )

        import sys

        sys.path.append("/opt/airflow")
        from helpers import (
            load_samples,
            autoencoder,
            custom_binary_crossentropy,
            dice_loss,
            generate_random_values,
            data_generator,
            f1_m,
        )

        with mlflow.start_run(run_name="train", parent_run_id=mlflow_parent_run_id):
            # Get hyperparameters from tuning run.
            mlflow.artifacts.download_artifacts(
                artifact_path="best_hyperparams.json",
                run_id=mlflow_run_id_deps[0],
                dst_path="/tmp",
            )
            with open("/tmp/best_hyperparams.json", "r") as f:
                hyperparameters = json.loads(f.read())

            def combined_loss(y_true, y_pred):
                return (1 - hyperparameters["alpha"]) * dice_loss(y_true, y_pred) + hyperparameters[
                    "alpha"
                ] * custom_binary_crossentropy(y_true, y_pred)

            if enabled:
                mlflow.artifacts.download_artifacts(
                    artifact_uri="s3://datasets/active-fires",
                    dst_path="/tmp/",
                )
                relative_path = "/tmp/active-fires"
                input_features = ["seviri_ir39_"]
                input_shape = (
                    128,
                    128,
                    len(input_features),
                )  # (tile_height, tile_width, num_features)

                model = autoencoder(
                    input_shape=input_shape,
                    filters=64,
                    activation="relu",
                    final_activation="sigmoid",
                )
                epochs = 300
                batch_size = 16
                optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparameters["learning_rate"])

                # Metrics
                recall = tf.keras.metrics.Recall()
                precision = tf.keras.metrics.Precision()
                metrics = ["accuracy", recall, precision, f1_m]

                model.compile(optimizer=optimizer, loss=combined_loss, metrics=metrics)
                print("Model compiled successfully.")

                import datetime

                file_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

                dir_path = os.path.join("/tmp/trained_models", file_datetime)
                os.makedirs(dir_path, exist_ok=True)

                model_weights_path = os.path.join(dir_path, "autoencoder_best.weights.h5")

                checkpoint = ModelCheckpoint(
                    filepath=model_weights_path,
                    monitor="val_loss",
                    verbose=1,
                    save_best_only=True,
                    save_weights_only=True,
                    mode="auto",
                )
                lr_reducer = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10, verbose=1, min_lr=1e-5)
                early_stopping = EarlyStopping(
                    monitor="val_loss",
                    patience=15,
                    verbose=1,
                    restore_best_weights=True,
                )

                callbacks_list = [checkpoint, lr_reducer, early_stopping]
                train_csv = load_samples(
                    os.path.join(relative_path, "train_dataset.csv"),
                    os.path.join(relative_path, "Tilled_Data"),
                )
                validation_csv = load_samples(
                    os.path.join(relative_path, "validation_dataset.csv"),
                    os.path.join(relative_path, "Tilled_Data"),
                )
                train_dataset = data_generator(train_csv, batch_size=batch_size, input_features=input_features)
                validation_dataset = data_generator(
                    validation_csv, batch_size=batch_size, input_features=input_features
                )

                print(f"Training tiles: {len(train_csv)}")
                print(f"Validation tiles: {len(validation_csv)}")

                model.fit(
                    train_dataset,
                    steps_per_epoch=len(train_csv) // batch_size,
                    epochs=epochs,
                    validation_data=validation_dataset,
                    validation_steps=len(validation_csv) // batch_size,
                    shuffle=True,
                    verbose=2,
                    callbacks=callbacks_list,
                )

                best_model = autoencoder()
                best_model.load_weights(model_weights_path)
            else:
                mlflow.artifacts.download_artifacts(
                    artifact_uri="s3://datasets/active-fires/saved_model",
                    dst_path="/tmp/",
                )
                best_model = mlflow.keras.load.load_model("/tmp/saved_model")

            mlflow.keras.save.log_model(best_model, "autoencoder_best_model")

            handle = mlflow.active_run().info.run_id
            return handle

    @task.kubernetes(
        name="evaluate",
        **dag_config.task_config_airflow.model_dump(),
        env_vars=dag_config.task_env_vars,
        env_from=dag_config.task_env_from_secrets,
    )
    def evaluate(mlflow_parent_run_id: str, mlflow_run_id_deps: List[str]):
        import mlflow

        mlflow.autolog()

        import os
        import numpy as np
        import pandas as pd

        import tensorflow as tf

        tf.random.set_seed(42)

        from sklearn.metrics import (
            recall_score,
            precision_score,
            f1_score,
            confusion_matrix,
        )

        import sys

        sys.path.append("/opt/airflow")
        from helpers import (
            load_samples,
            data_generator,
        )

        def to_labels(pos_probs, threshold):
            """Convert probabilities to binary labels based on a threshold."""
            return (pos_probs >= threshold).astype("int")

        def compute_metrics(y_pred, y_true, predefined_threshold=None):
            """
            Compute evaluation metrics including recall, precision, F1-score, IOU, and confusion matrix.
            Optionally calculate optimal threshold if none is predefined.
            """
            print("Computing metrics ...")
            y_true_flat = y_true.flatten()
            results = []

            # Determine the optimal threshold if none is predefined
            if predefined_threshold is None:
                thresholds = np.arange(0, 1, 0.01)
                f1_scores = [f1_score(y_true_flat, to_labels(y_pred.flatten(), t)) for t in thresholds]
                optimal_idx = np.argmax(f1_scores)
                threshold = thresholds[optimal_idx]
                print(f"Optimal Threshold = {threshold:.3f}, F1-Score = {f1_scores[optimal_idx]:.5f}")
            else:
                threshold = predefined_threshold

            # Calculate metrics
            labels = to_labels(y_pred.flatten(), threshold)
            recall = recall_score(y_true_flat, labels)
            precision = precision_score(y_true_flat, labels)
            f1_score_value = f1_score(y_true_flat, labels)
            cmatrix = confusion_matrix(y_true_flat, labels)
            tn, fp, fn, tp = cmatrix.ravel()
            iou = tp / (tp + fn + fp)

            # Log metrics
            print(f"Recall: {recall:.5f}, Precision: {precision:.5f}, F1-Score: {f1_score_value:.5f}")
            print(f"IOU: {iou:.5f}, TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

            # Save metrics to a dictionary
            metrics = {
                "recall": recall,
                "precision": precision,
                "f1_score": f1_score_value,
                "IOU": iou,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
            }
            results.append(metrics)
            metrics_df = pd.DataFrame(results)

            # Return metrics and threshold if no predefined threshold was given
            return (threshold, metrics_df) if predefined_threshold is None else metrics_df

        def evaluate_class(
            model,
            test_data_path,
            relative_path,
            input_features,
            class_id,
            threshold=None,
        ):
            print(f"Evaluating metrics for Class {class_id} ...")

            test_csv = load_samples(test_data_path, rel_path=relative_path, class_id=class_id)
            batch_size = len(test_csv)

            print(f"Number of tiles: {batch_size}")
            test_datagen = data_generator(test_csv, batch_size=batch_size, input_features=input_features)
            x, y = next(test_datagen)

            predictions = model.predict(x, verbose=1, steps=1)
            metrics_df = compute_metrics(predictions, y, predefined_threshold=threshold)

            if class_id == 1:
                metrics_df["Class"] = "small"
            elif class_id == 0:
                metrics_df["Class"] = "medium"
            elif class_id == -1:
                metrics_df["Class"] = "large"
            metrics_df["Num_Tiles"] = batch_size

            return metrics_df

        with mlflow.start_run(run_name="evaluate", parent_run_id=mlflow_parent_run_id):
            model = mlflow.keras.load.load_model(f"runs:/{mlflow_run_id_deps[0]}/autoencoder_best_model")

            mlflow.artifacts.download_artifacts(
                artifact_uri="s3://datasets/active-fires",
                dst_path="/tmp/",
            )
            dataset_path = "/tmp/active-fires"
            test_csv_path = os.path.join(dataset_path, "test_dataset.csv")
            test_csv = load_samples(test_csv_path, os.path.join(dataset_path, "Tilled_Data"))

            input_features = ["seviri_ir39_"]
            test_datagen = data_generator(test_csv, batch_size=len(test_csv), input_features=input_features)
            x, y = next(test_datagen)
            predictions = model.predict(x, verbose=1, steps=1)

            optimal_threshold, whole_metrics_df = compute_metrics(predictions, y)
            whole_metrics_df["Class"] = "All"
            whole_metrics_df["Num_Tiles"] = len(test_csv)

            small_metrics_df = evaluate_class(
                model,
                test_csv_path,
                os.path.join(dataset_path, "Tilled_Data"),
                input_features,
                class_id=1,
                threshold=optimal_threshold,
            )
            medium_metrics_df = evaluate_class(
                model,
                test_csv_path,
                os.path.join(dataset_path, "Tilled_Data"),
                input_features,
                class_id=0,
                threshold=optimal_threshold,
            )
            large_metrics_df = evaluate_class(
                model,
                test_csv_path,
                os.path.join(dataset_path, "Tilled_Data"),
                input_features,
                class_id=-1,
                threshold=optimal_threshold,
            )

            final_metrics = pd.concat(
                [
                    whole_metrics_df,
                    small_metrics_df,
                    medium_metrics_df,
                    large_metrics_df,
                ],
                axis=0,
            )
            mlflow.log_table(final_metrics, artifact_file="autoencoder_best_model_metrics.parquet")

    master_run_id = entrypoint()
    tune_rid = tune(mlflow_parent_run_id=master_run_id, enabled=tune_enabled)
    train_rid = train(mlflow_parent_run_id=master_run_id, mlflow_run_id_deps=[tune_rid], enabled=train_enabled)
    evaluate_rid = evaluate(mlflow_parent_run_id=master_run_id, mlflow_run_id_deps=[train_rid])


active_fires_model_development_dag(**dag_config.runtime_config.model_dump())
