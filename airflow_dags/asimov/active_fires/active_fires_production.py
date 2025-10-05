from airflow.decorators import task, dag
from airflow.models import Variable

from asimov.config import ActiveFiresProductionConfig

dag_config = ActiveFiresProductionConfig.model_validate_json(Variable.get("af-production-config"))


@task.kubernetes(
    name="fetch_products",
    **dag_config.task_config_airflow.model_dump(),
    env_vars=dag_config.task_env_vars,
    env_from=dag_config.task_env_from_secrets,
)
def fetch_products(start_time, end_time):
    import eumdac
    import os
    from datetime import datetime
    import requests

    token = eumdac.AccessToken([os.environ["CONSUMER_KEY"], os.environ["CONSUMER_SECRET"]])
    datastore = eumdac.DataStore(token)
    selected_collection = datastore.get_collection("EO:EUM:DAT:MSG:MSG15-RSS")

    try:
        products = selected_collection.search(
            sat="MSG4", dtstart=datetime.fromisoformat(start_time), dtend=datetime.fromisoformat(end_time)
        )
        print(f"Found {products.total_results} datasets for satellite type MSG4")
        return [product.metadata["properties"] for product in products] if products.total_results > 0 else []

    except (eumdac.collection.CollectionError, requests.exceptions.RequestException) as error:
        print(f"Error fetching datasets: {error}")
        exit(1)


@task.kubernetes(
    name="show_product_metadata",
    **dag_config.task_config_airflow.model_dump(),
    env_vars=dag_config.task_env_vars,
    env_from=dag_config.task_env_from_secrets,
)
def process_products(products):
    import os
    from tqdm import tqdm
    import eumdac

    def copy_with_progress(product_id, destination_dir, chunk_size=128 * 1024):
        token = eumdac.AccessToken([os.environ["CONSUMER_KEY"], os.environ["CONSUMER_SECRET"]])
        datastore = eumdac.DataStore(token)
        product_ = datastore.get_product("EO:EUM:DAT:MSG:MSG15-RSS", product_id)

        file_size = product_.size
        file_path = os.path.join(destination_dir, f"{product_id}.zip")

        with tqdm(
            total=file_size, unit="B", unit_scale=True, desc=f"Downloading {product_id}", bar_format="{percentage:.1f}%"
        ) as pbar:
            with product_.open() as source, open(file_path, "wb") as destination:
                while True:
                    buffer = source.read(chunk_size)
                    if not buffer:
                        break
                    destination.write(buffer)
                    pbar.update(len(buffer))

        return file_path

    def process_single_product(data_dir, product, results_dir):
        from zipfile import ZipFile
        from osgeo import gdal
        import numpy as np
        import requests
        import os

        product_zip_path = os.path.join(data_dir, f"{product['identifier']}.zip")
        result_file_tiff = os.path.join(
            results_dir, os.path.basename(product_zip_path).replace(".zip", "_WGS84_active_fires.tiff")
        )
        product_zip_path = copy_with_progress(product["identifier"], data_dir)
        with ZipFile(product_zip_path) as zip_ref:
            zip_ref.extractall(data_dir)
        input_file_nat = product_zip_path.replace(".zip", ".nat")
        output_file_tiff = product_zip_path.replace(".zip", "_WGS84.tiff")
        seviri_data = gdal.Open(input_file_nat)
        gdal.Warp(
            output_file_tiff,
            seviri_data,
            dstSRS='GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],UNIT["degree",0.0174532925199433]]',
            outputBounds=(20.15, 34.92, 29.62, 41.85),
            xRes=0.008333333333333333,
            yRes=-0.008333333333333333,
            resampleAlg=gdal.gdalconst.GRA_Bilinear,
            srcNodata=32635,
            creationOptions=["COMPRESS=DEFLATE", "INTERLEAVE=BAND", "PREDICTOR=2"],
        )
        input_seviri = gdal.Open(output_file_tiff)
        input_data = input_seviri.ReadAsArray()[3] / 1000.0
        ext_array = input_data[np.newaxis, ..., np.newaxis]
        print(ext_array.shape)

        import io

        buffer = io.BytesIO()
        np.save(buffer, ext_array)
        buffer.seek(0)
        response = requests.post(
            os.environ["MODEL_DEPLOYMENT_URL"],
            data=buffer.getvalue(),
            headers={"Content-Type": "application/octet-stream"},
        )
        response.raise_for_status()
        response_buffer = io.BytesIO(response.content)
        prediction = np.load(response_buffer)
        prediction = np.squeeze(prediction)
        out_data = gdal.GetDriverByName("GTiff").Create(
            result_file_tiff, input_data.shape[1], input_data.shape[0], 1, gdal.GDT_Float32
        )
        out_data.SetGeoTransform(input_seviri.GetGeoTransform())
        out_data.SetProjection(input_seviri.GetProjection())
        out_data.GetRasterBand(1).WriteArray(prediction)
        out_data.FlushCache()
        print(f"Saved active fire probabilities to {result_file_tiff}")

    data_dir, results_dir = "/tmp/seviri-data", "/tmp/results"
    os.makedirs(data_dir)
    os.makedirs(results_dir)

    for product in products:
        process_single_product(data_dir, product, results_dir)

    import mlflow
    with mlflow.start_run(run_name="active-fires-production"):
        mlflow.log_artifacts(results_dir)


@dag(tags=["asimov", "active-fires"])
def active_fires_production_dag(start_time: str, end_time: str):
    products = fetch_products(start_time, end_time)
    process_products(products)


dag_instance = active_fires_production_dag("2024-08-12T23:45:00", "2024-08-12T23:59:00")
