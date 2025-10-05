from airflow.decorators import task, dag
from airflow.models import Variable
from asimov.config import Config

dag_config = Config.model_validate_json(Variable.get("precp-production-config"))


@task.kubernetes(
    name="precipitation-infer",
    **dag_config.task_config_airflow.model_dump(),
    env_vars=dag_config.task_env_vars,
    env_from=dag_config.task_env_from_secrets,
)
def infer(model_version, start_date, end_date):
    import mlflow

    with mlflow.start_run(run_name="precipitation-model-prod"):
        import os
        import pandas as pd
        import numpy as np
        import rasterio
        from datetime import datetime, timedelta
        import sys

        sys.path.append("/opt/airflow")

        mlflow.artifacts.download_artifacts(
            artifact_uri=f"s3://datasets/precipitation",
            dst_path="/tmp/",
        )
        base_dir = "/tmp/precipitation"
        predictions_folder = "/tmp/predictions"
        os.makedirs(predictions_folder, exist_ok=True)

        static_paths = {
            "Value TWI": os.path.join(base_dir, "Statics_1000m", "TWI_res.tif"),
            "Value TPI": os.path.join(
                base_dir, "Statics_1000m", "srtmMtpiGreece_res.tif"
            ),
            "Value ASPECT": os.path.join(
                base_dir, "Statics_1000m", "ASPECT_greece_res.tif"
            ),
            "Value landformsGreece": os.path.join(
                base_dir, "Statics_1000m", "landformsGreece_res.tif"
            ),
            "Value topoDiversity": os.path.join(
                base_dir, "Statics_1000m", "topoDiversity_res.tif"
            ),
        }
        dynamic_folders = {
            "Band 1": os.path.join(base_dir, "Wind_resampled_combined2"),
            "Band 2": os.path.join(base_dir, "Wind_resampled_combined2"),
            "Band 3": os.path.join(base_dir, "Wind_resampled_combined2"),
            "Band 4": os.path.join(base_dir, "Wind_resampled_combined2"),
            "Value WVC": os.path.join(base_dir, "resampled_water_vapor"),
            "Value IMERG": os.path.join(base_dir, "resampled_imerg_1000m_aggregated"),
            "Value CER": os.path.join(base_dir, "resampled_modis_combined", "CER"),
            "Value COT": os.path.join(base_dir, "resampled_modis_combined", "COT"),
            "Value CWP": os.path.join(base_dir, "resampled_modis_combined", "CWP"),
        }

        model = mlflow.sklearn.load_model(
            f"models:/precipitation-inference/{model_version}"
        )

        def read_geotiff(path):
            with rasterio.open(path) as src:
                return src.read(1), src.transform, src.crs

        def get_dynamic_paths(date):
            year, month, day, date_str, doy = (
                date.strftime("%Y"),
                date.strftime("%m"),
                date.strftime("%d"),
                date.strftime("%Y%m%d"),
                date.strftime("%j"),
            )

            return {
                "Band 1": os.path.join(
                    dynamic_folders["Band 1"], f"{date_str}_wind_res", "band_1.tif"
                ),
                "Band 2": os.path.join(
                    dynamic_folders["Band 2"], f"{date_str}_wind_res", "band_2.tif"
                ),
                "Band 3": os.path.join(
                    dynamic_folders["Band 3"], f"{date_str}_wind_res", "band_3.tif"
                ),
                "Band 4": os.path.join(
                    dynamic_folders["Band 4"], f"{date_str}_wind_res", "band_4.tif"
                ),
                "Value WVC": os.path.join(
                    dynamic_folders["Value WVC"],
                    year,
                    month,
                    f"ColumnWV_{year}_{month}_{day}.tif",
                ),
                "Value IMERG": os.path.join(
                    dynamic_folders["Value IMERG"],
                    f"IMERG_aggregated_{date_str}_1000m.tif",
                ),
                "Value CER": os.path.join(
                    dynamic_folders["Value CER"], f"cer_res_A{year}{doy}.tif"
                ),
                "Value COT": os.path.join(
                    dynamic_folders["Value COT"], f"cot_res_A{year}{doy}.tif"
                ),
                "Value CWP": os.path.join(
                    dynamic_folders["Value CWP"], f"cwp_res_A{year}{doy}.tif"
                ),
            }

        def predict_and_save(date):
            dynamic_paths = get_dynamic_paths(date)
            paths = {**static_paths, **dynamic_paths}
            data, transform, crs, original_shape = {}, None, None, None

            # Check if wind data exists for the date
            wind_path_exists = True
            for band_key in ["Band 1", "Band 2", "Band 3", "Band 4"]:
                if not os.path.exists(paths[band_key]):
                    wind_path_exists = False
                    break

            if not wind_path_exists and date == datetime(2010, 1, 1):
                print(f"Wind data missing for 2010-01-01. Skipping...")
                return  # Skip the entire prediction

            for key, path in paths.items():
                try:
                    array, transform, crs = read_geotiff(path)
                    if original_shape is None:
                        original_shape = array.shape
                    data[key] = array.flatten()
                except FileNotFoundError:
                    print(f"Missing file: {path}. Skipping...")
                    return
                except Exception as e:
                    print(f"Error reading {path}: {e}")
                    return

            df_inference = pd.DataFrame(data)

            # Extract month from the IMERG filename
            imerg_filename = os.path.basename(
                paths["Value IMERG"]
            )  # Get the filename (e.g., "IMERG_aggregated_20240601_1000m.tif")
            date_str = imerg_filename.split("_")[
                2
            ]  # Extract the date part (e.g., "20240601")
            month = int(date_str[4:6])  # Extract the month (e.g., "06" -> 6)

            # Add the month feature to the DataFrame
            df_inference["month"] = month

            # Apply feature engineering
            df_inference["COT_CER_Ratio"] = np.where(
                df_inference["Value CER"] > 1e-6,
                df_inference["Value COT"] / df_inference["Value CER"],
                0,  # Default value when Value CER is too small
            )
            df_inference["CWP_Squared"] = df_inference["Value CWP"] ** 2

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
                "month",
                "COT_CER_Ratio",
                "CWP_Squared",
            ]

            # Prepare input data
            X_inference = df_inference[features]
            X_inference.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Predict using the trained model
            y_pred_log = model.predict(X_inference)
            y_pred = np.maximum(
                np.expm1(y_pred_log), 0
            )  # Ensure non-negative predictions!!!!

            # Reshape predictions to original GeoTIFF shape
            prediction_array = np.maximum(
                y_pred.reshape(original_shape), 0
            )  # Clip to non-negative values
            output_path = os.path.join(
                predictions_folder,
                f"predicted_precipitation_{date.strftime('%Y%m%d')}.tif",
            )

            with rasterio.open(
                output_path,
                "w",
                driver="GTiff",
                height=prediction_array.shape[0],
                width=prediction_array.shape[1],
                count=1,
                dtype=prediction_array.dtype,
                crs=crs,
                transform=transform,
            ) as dst:
                dst.write(prediction_array, 1)

            print(f"Prediction saved to {output_path}")

        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

        while start_date_obj <= end_date_obj:
            print(f"Processing date: {start_date_obj.strftime('%Y-%m-%d')}")
            predict_and_save(start_date_obj)
            start_date_obj += timedelta(days=1)

        mlflow.log_artifacts(predictions_folder, "results")


@dag(tags=["asimov", "precipitation"])
def precipitation_model_production_dag(
    model_version: int, start_date: str, end_date: str
):
    infer(model_version, start_date, end_date)


dag_instance = precipitation_model_production_dag(1, "2023-08-12", "2023-08-14")
