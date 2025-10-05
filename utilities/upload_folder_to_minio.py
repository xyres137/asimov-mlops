from minio import Minio

import os
import glob

client = Minio(
    "mlflow-minio.mlflow:80",
    access_key="",
    secret_key="",
    secure=False,
)


def upload_local_directory_to_minio(local_path, bucket_name, minio_path):
    assert os.path.isdir(local_path)

    for local_file in glob.glob(local_path + "/**"):
        local_file = local_file.replace(os.sep, "/")  # Replace \ with / on Windows
        if not os.path.isfile(local_file):
            upload_local_directory_to_minio(local_file, bucket_name, minio_path + "/" + os.path.basename(local_file))
        else:
            remote_path = os.path.join(minio_path, local_file[1 + len(local_path) :])
            remote_path = remote_path.replace(os.sep, "/")  # Replace \ with / on Windows
            client.fput_object(bucket_name, remote_path, local_file)


upload_local_directory_to_minio("/Users/dl/code/mlops/active_fires_dataset", "datasets", "active-fires")
