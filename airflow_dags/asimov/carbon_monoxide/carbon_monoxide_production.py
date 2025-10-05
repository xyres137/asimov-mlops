from airflow.decorators import task, dag
from airflow.models import Variable

from asimov.config import CarbonMonoxideProductionConfig

dag_config = CarbonMonoxideProductionConfig.model_validate_json(Variable.get("cm-production-config"))


@dag(tags=["asimov", "carbon-monoxide"])
def carbon_monoxide_production_dag(date: str):
    @task.kubernetes(
        name="predict_co",
        **dag_config.task_config_airflow.model_dump(),
        env_vars=dag_config.task_env_vars,
        env_from=dag_config.task_env_from_secrets,
    )
    def predict(request_date: str):
        import os

        DOWNLOAD_DIR = "/tmp/downloads"
        RESAMPLING_DIR = "/tmp/resampling"
        RESULTS_DIR = "/tmp/results"

        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        os.makedirs(RESAMPLING_DIR, exist_ok=True)

        QA_THRESHOLD = 0.5

        import requests
        import subprocess
        from zipfile import ZipFile
        import xarray
        from osgeo import gdal
        import shutil
        import pandas as pd
        import earthaccess
        import numpy as np

        np.int = np.int32
        np.float = np.float64
        np.bool = np.bool_

        from haversine import haversine
        import math

        def query(name, date, extent=None):
            base_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

            filters = [
                f"contains(Name, '{name}')",
                f"ContentDate/Start gt {date}T05:00:00.000Z",
                f"ContentDate/End lt {date}T16:59:59.000Z",
            ]
            if extent:
                filters.append(f"OData.CSC.Intersects(area=geography'SRID=4326;{extent}')")

            filter_query = " and ".join(filters)
            url = f"{base_url}?$filter={filter_query}"

            resp = requests.get(url)
            resp.raise_for_status()

            return resp.json()

        def get_keycloak_token(username, password):
            response = requests.post(
                "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
                data={
                    "client_id": "cdse-public",
                    "username": username,
                    "password": password,
                    "grant_type": "password",
                },
            )
            response.raise_for_status()
            return response.json()["access_token"]

        def download_file(token, file_url, dest_path):
            curl_command = [
                "curl",
                "-H",
                f"Authorization: Bearer {token}",
                f"{file_url}",
                "--location-trusted",
                "--limit-rate",
                "9M",
                "-o",
                f"{dest_path}",
            ]

            subprocess.run(curl_command, check=True, capture_output=True)

        def download_modis_ndvi(impact_time, path, new_name, extent):
            print("Requesting MODIS NDVI data ...")

            earthaccess.login(strategy="environment")
            results = earthaccess.search_data(
                short_name="MOD13C2",
                bounding_box=extent,  # [min_lon, min_lat, max_lon, max_lat]
                temporal=(impact_time, impact_time),  # Single day search
            )
            files = earthaccess.download(results, path)
            if not files:
                print("No files were downloaded")
                return

            print(f"Downloaded {len(results)} files to {path}")
            for file in os.listdir(path):
                if "MOD13C2" in file:
                    original_path = os.path.join(path, file)
                    new_path = os.path.join(new_name)
                    os.rename(original_path, new_path)
                    print(f"Renamed file {original_path} to {new_path}")

        def ERA5_Land_processor(filename, outpath, bounds):
            ds_era5 = gdal.Open(filename, gdal.gdalconst.GA_ReadOnly)

            era5_datasets = [
                "2m_dewpoint_temperature",
                "2m_temperature",
                "surface_net_solar_radiation",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "surface_pressure",
                "total_precipitation",
            ]

            era5_datasets = ds_era5.GetSubDatasets()

            for k, dataset in era5_datasets:
                print(k, dataset)

                src = gdal.Open(k, gdal.gdalconst.GA_ReadOnly)

                options = gdal.WarpOptions(
                    xRes=0.008333333333333333218,
                    yRes=0.008333333333333333218,
                    resampleAlg=gdal.gdalconst.GRA_Bilinear,
                    srcNodata=-32767,
                    dstSRS="EPSG:4326",
                    outputBounds=bounds,
                )
                g = gdal.Warp(f"{outpath.split('.')[0]}_{k.split(':')[2]}.tiff", src, options=options)

                g = None

            era5_datasets = None
            ds_era5 = None

        def find_impact_coords(lonext, latext, bbox):
            import numpy as np
            from matplotlib import path

            bbox = np.array(bbox)
            mypath = np.array([bbox[[0, 1, 1, 0]], bbox[[2, 2, 3, 3]]]).T
            p = path.Path(mypath)

            points = np.vstack((lonext.flatten(), latext.flatten())).T
            n, m = np.shape(lonext)

            inside = p.contains_points(points).reshape((n, m))
            ii, jj = np.meshgrid(range(m), range(n))

            if np.size(ii[inside]) == 0:
                return 0, 0, 0, 0

            return min(ii[inside]), max(ii[inside]), min(jj[inside]), max(jj[inside])

        def S5P_CO_processor(work_dir, filename, dstname, date, output_bounds):
            # 1. Load lat and lon data for preparation of making VRTs
            ds_product = xarray.open_dataset(filename, group="PRODUCT", cache=True)
            co_tropomi = ds_product["carbonmonoxide_total_column"].sel(time=date)
            co_tropomi = co_tropomi.where(ds_product["qa_value"].sel(time=date) > QA_THRESHOLD)

            lon = ds_product["longitude"].sel(time=date).values
            lat = ds_product["latitude"].sel(time=date).values

            idxi0, idxi1, idxj0, idxj1 = find_impact_coords(lon[:], lat[:], output_bounds)

            # get raster dimension
            ysize, xsize = lon[idxj0:idxj1, idxi0:idxi1].shape

            # Save the lat/lon to TIFFs and VRTs
            driver = gdal.GetDriverByName("GTiff")
            dataset = driver.Create(
                os.path.join(work_dir, "lon.tif"),
                xsize,
                ysize,
                1,
                gdal.GDT_Float64,
            )

            band = dataset.GetRasterBand(1).WriteArray(lon[idxj0:idxj1, idxi0:idxi1])
            dataset = None

            dataset = driver.Create(
                os.path.join(work_dir, "lat.tif"),
                xsize,
                ysize,
                1,
                gdal.GDT_Float64,
            )

            band = dataset.GetRasterBand(1).WriteArray(lat[idxj0:idxj1, idxi0:idxi1])
            dataset = None
            lat = None
            lon = None

            # *-- Construct and save VRTs --*
            lon_vrt = f"""<VRTDataset rasterXSize="{xsize}" rasterYSize="{ysize}">
                  <SRS>GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9108"]],AUTHORITY["EPSG","4326"]]</SRS>
              <VRTRasterBand dataType="Float64" band="1">
                <SimpleSource>
                  <SourceFilename relativeToVRT="1">lon.tif</SourceFilename>
                  <SourceBand>1</SourceBand>
                  <SourceProperties RasterXSize="{xsize}" RasterYSize="{ysize}" DataType="Float32" BlockXSize="256" BlockYSize="256" />
                  <SrcRect xOff="0" yOff="0" xSize="{xsize}" ySize="{ysize}" />
                  <DstRect xOff="0" yOff="0" xSize="{xsize}" ySize="{ysize}" />
                </SimpleSource>
              </VRTRasterBand>
            </VRTDataset>"""

            lat_vrt = f"""<VRTDataset rasterXSize="{xsize}" rasterYSize="{ysize}">
                      <SRS>GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9108"]],AUTHORITY["EPSG","4326"]]</SRS>
                  <VRTRasterBand dataType="Float64" band="1">
                    <SimpleSource>
                      <SourceFilename relativeToVRT="1">lat.tif</SourceFilename>
                      <SourceBand>1</SourceBand>
                      <SourceProperties RasterXSize="{xsize}" RasterYSize="{ysize}" DataType="Float32" BlockXSize="256" BlockYSize="256" />
                      <SrcRect xOff="0" yOff="0" xSize="{xsize}" ySize="{ysize}" />
                      <DstRect xOff="0" yOff="0" xSize="{xsize}" ySize="{ysize}" />
                    </SimpleSource>
                  </VRTRasterBand>
                </VRTDataset>"""

            # save VRT files now
            with open(os.path.join(work_dir, "lon.vrt"), "w") as text_file:
                text_file.write(lon_vrt)
            with open(os.path.join(work_dir, "lat.vrt"), "w") as text_file:
                text_file.write(lat_vrt)

            # Pull out the data we need

            dataset = driver.Create(
                os.path.join(work_dir, "data.tif"),
                xsize,
                ysize,
                1,
                gdal.GDT_Float64,
            )

            band = dataset.GetRasterBand(1).WriteArray(co_tropomi.values[idxj0:idxj1, idxi0:idxi1])
            dataset = None

            vrt_raster_band_tmpl = f"""<VRTRasterBand dataType="Float64" band="1">
                    <SimpleSource>
                      <SourceFilename relativeToVRT="1">data.tif</SourceFilename>
                      <SourceBand>1</SourceBand>
                      <SourceProperties RasterXSize="{xsize}" RasterYSize="{ysize}" DataType="Float32" BlockXSize="256" BlockYSize="256" />
                      <SrcRect xOff="0" yOff="0" xSize="{xsize}" ySize="{ysize}" />
                      <DstRect xOff="0" yOff="0" xSize="{xsize}" ySize="{ysize}" />
                    </SimpleSource>
                    </VRTRasterBand>
                    """

            lon_file = os.path.join(work_dir, "lon.tif")
            lat_file = os.path.join(work_dir, "lat.tif")
            vrt_main_tmpl = f"""<VRTDataset rasterXSize="{xsize}" rasterYSize="{ysize}">
                     <metadata domain="GEOLOCATION">
                       <mdi key="X_DATASET">{lon_file}</mdi>
                       <mdi key="X_BAND">1</mdi>
                       <mdi key="Y_DATASET">{lat_file}</mdi>
                       <mdi key="Y_BAND">1</mdi>
                       <mdi key="PIXEL_OFFSET">0</mdi>
                       <mdi key="LINE_OFFSET">0</mdi>
                       <mdi key="PIXEL_STEP">1</mdi>
                       <mdi key="LINE_STEP">1</mdi>
                     </metadata>
                         {vrt_raster_band_tmpl}
                  </VRTDataset>"""

            with open(os.path.join(work_dir, "data.vrt"), "w") as text_file:
                text_file.write(vrt_main_tmpl)

            g = gdal.Warp(
                dstname,
                os.path.join(work_dir, "data.vrt"),
                dstSRS="EPSG:4326",
                xRes=0.008333333333333333218,
                yRes=0.008333333333333333218,
                resampleAlg=gdal.gdalconst.GRA_Bilinear,
                geoloc=True,
                srcNodata=9.96921e36,
                outputBounds=output_bounds,
            )

            g = None

            # Clean up temporary files
            temp_files = ["lon.tif", "lat.tif", "data.tif", "lon.vrt", "lat.vrt", "data.vrt"]
            for file in temp_files:
                os.remove(os.path.join(work_dir, file))

            print(f"Processing complete. Output saved to {dstname}")

        def get_time_index(longitude, latitude, target_lon, target_lat):
            import numpy as np

            R = 6371000  # radius of earth in meters
            latitude2 = latitude[:]
            longitude2 = longitude[:]
            lat1 = np.radians(target_lat)
            lat2 = np.radians(latitude2)

            delta_lat = np.radians(latitude2 - target_lat)
            delta_lon = np.radians(longitude2 - target_lon)
            a = (np.sin(delta_lat / 2)) * (np.sin(delta_lat / 2)) + (np.cos(lat1)) * (np.cos(lat2)) * (
                np.sin(delta_lon / 2)) * (
                    np.sin(delta_lon / 2)
                )
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            d = R * c
            x2, y2 = np.unravel_index(d.argmin(), d.shape)
            return x2, y2

        from datetime import datetime
        from dateutil import parser
        import joblib

        # Configuration - User defined
        request_datetime = datetime.strptime(request_date, "%Y%m%d")

        lon_bounds = (20.15, 29.608333333333334)  # minLon, maxLon
        lat_bounds = (34.925, 41.83333333333333)  # minLat, maxLat
        extent = [
            lat_bounds[1],
            lon_bounds[0],
            lat_bounds[0],
            lon_bounds[1],
        ]  # [N, W, S, E]
        output_bounds = (lon_bounds[0], lat_bounds[0], lon_bounds[1], lat_bounds[1])
        bbox = f"POLYGON(({lon_bounds[0]} {lat_bounds[0]},{lon_bounds[0]} {lat_bounds[1]},{lon_bounds[1]} {lat_bounds[1]},{lon_bounds[1]} {lat_bounds[0]},{lon_bounds[0]} {lat_bounds[0]}))"

        import mlflow

        mlflow.artifacts.download_artifacts(
            artifact_uri="s3://datasets/carbon-monoxide/std_scaler.bin",
            dst_path=DOWNLOAD_DIR,
        )
        scaler = joblib.load(os.path.join(DOWNLOAD_DIR, "std_scaler.bin"))

        # Download Sentinel-5P Carbon Monoxide
        os.makedirs(
            os.path.join(DOWNLOAD_DIR, "Sentinel5P_CO", request_datetime.strftime("%Y-%m-%d")),
            exist_ok=True,
        )
        # [30/4/2018, 25/7/2022]: RPRO | [26/7/2022, ...]: OFFL
        sentinel_cat = "OFFL" if request_datetime >= parser.parse("20220726") else "RPRO"
        results = query(
            name=f"S5P_{sentinel_cat}_L2__CO",
            date=request_datetime.strftime("%Y-%m-%d"),
            extent=bbox,
        )

        if len(results["value"]) == 0:
            print("No data for this date")
            exit()

        for i, item in enumerate(results["value"]):
            version = item["Name"].split("_")[12]
            login, password = os.environ["CDSE_LOGIN"], os.environ["CDSE_PASSWORD"]
            token = get_keycloak_token(login, password)

            s5p_path = os.path.join(DOWNLOAD_DIR, "Sentinel5P_CO")
            if version == "03":
                print(f"Downloading :", item["Name"])
                file_url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({item['Id']})/$value"

                # Since 1/1/2019 - zip file
                if request_datetime < parser.parse("20190101"):
                    zip_path = os.path.join(
                        s5p_path,
                        request_datetime.strftime("%Y-%m-%d"),
                        f"{item['Name'].replace('.nc', '.zip')}",
                    )
                    download_file(token, file_url, zip_path)

                    with ZipFile(zip_path) as zObject:
                        zObject.extractall(os.path.join(s5p_path, request_datetime.strftime("%Y-%m-%d")))

                    shutil.move(
                        os.path.join(
                            s5p_path,
                            request_datetime.strftime("%Y-%m-%d"),
                            item["Name"].split(".")[0],
                            item["Name"],
                        ),
                        os.path.join(
                            s5p_path,
                            request_datetime.strftime("%Y-%m-%d"),
                            item["Name"],
                        ),
                    )
                    shutil.rmtree(
                        os.path.join(
                            s5p_path,
                            request_datetime.strftime("%Y-%m-%d"),
                            item["Name"].split(".")[0],
                        )
                    )
                    os.remove(zip_path)
                else:
                    nc_path = os.path.join(
                        s5p_path,
                        request_datetime.strftime("%Y-%m-%d"),
                        f"{item['Name']}",
                    )
                    download_file(token, file_url, nc_path)

        era5_path = os.path.join(DOWNLOAD_DIR, "ERA5_Land")
        os.makedirs(era5_path, exist_ok=True)

        era_filename = "ERA5_Land_" + request_datetime.strftime("%Y%m%d") + ".nc"
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"s3://datasets/carbon-monoxide/ERA5_Land/{era_filename}",
            dst_path=era5_path,
        )
        era5_file = os.path.join(era5_path, era_filename)

        modis_path = os.path.join(DOWNLOAD_DIR, "MODIS_VEG")
        os.makedirs(modis_path, exist_ok=True)
        modis_file = os.path.join(modis_path, "MOD13C2_" + request_datetime.strftime("%Y%m%d") + ".nc")
        download_modis_ndvi(request_datetime.strftime("%Y-%m"), modis_path, modis_file, output_bounds)

        # Resample Datasets
        resampled_modis = os.path.join(RESAMPLING_DIR, f"Resampled_MOD13C2_{request_datetime.strftime('%Y-%m')}.tiff")

        if os.path.exists(modis_file) and not os.path.exists(resampled_modis):
            ds_modis = gdal.Open(modis_file, gdal.gdalconst.GA_ReadOnly)
            modis_datasets = ds_modis.GetSubDatasets()

            getBand = gdal.Open(modis_datasets[0][0])

            if "CMG 0.05 Deg Monthly NDVI" in getBand.GetDescription():
                options = gdal.WarpOptions(
                    xRes=0.008333333333333333218,
                    yRes=0.008333333333333333218,
                    resampleAlg=gdal.gdalconst.GRA_Bilinear,
                    srcNodata=-32767,
                    dstSRS="EPSG:4326",
                    outputBounds=output_bounds,
                )
                g = gdal.Warp(resampled_modis, getBand, options=options)

                g = None

            getBand = None
            ds_modis = None

        # 2 Resampled ERA5
        resampled_era5_file = os.path.join(RESAMPLING_DIR,
                                           f"Resampled_ERA5_Land_{request_datetime.strftime('%Y-%m-%d')}.tiff")

        ERA5_Land_processor(era5_file, resampled_era5_file, output_bounds)

        # 3 Resample Sentinel-5P
        for S5P_file in os.listdir(os.path.join(s5p_path, request_datetime.strftime("%Y-%m-%d"))):
            if S5P_file.startswith("S5P_"):
                print(f"Processing file: {S5P_file}\n")

                datafile = os.path.join(s5p_path, request_datetime.strftime("%Y-%m-%d"), S5P_file)

                orbit = xarray.open_dataset(datafile).orbit
                product_version = xarray.open_dataset(datafile).product_version
                algorithm_version = xarray.open_dataset(datafile).algorithm_version
                coverage_start = xarray.open_dataset(datafile).time_coverage_start
                cd = request_datetime.strftime("%Y-%m-%d")

                ds_PRODUCT = xarray.open_dataset(datafile, group="PRODUCT", cache=True)
                lon_in = ds_PRODUCT["longitude"].sel(time=cd).values
                lat_in = ds_PRODUCT["latitude"].sel(time=cd).values
                impact_delta = ds_PRODUCT["delta_time"].values[:]
                time_utc = ds_PRODUCT["time_utc"].values[:]

                idxi0, idxi1, idxj0, idxj1 = find_impact_coords(
                    lon_in[:], lat_in[:], [lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]]
                )

                if idxi0 == idxi1:
                    print("Orbit is out of range. Skipping...")
                    continue

                ds_PRODUCT = None

                resampled_s5p_path = os.path.join(
                    RESAMPLING_DIR, f"Resampled_CO_{request_datetime.strftime('%Y-%m-%d')}_{orbit}.tiff"
                )
                print(resampled_s5p_path)
                S5P_CO_processor(
                    RESAMPLING_DIR, datafile, resampled_s5p_path, request_datetime.strftime("%Y-%m-%d"), output_bounds
                )

                # Read Resampled Files
                s5p_src = gdal.Open(resampled_s5p_path)
                band = s5p_src.GetRasterBand(1)
                co_tropomi_value = band.ReadAsArray()

                mlflow.artifacts.download_artifacts(
                    artifact_uri="s3://datasets/carbon-monoxide/Greece_TPI.tif",
                    dst_path=DOWNLOAD_DIR,
                )
                tpi_src = gdal.Open(os.path.join(DOWNLOAD_DIR, "Greece_TPI.tif"), gdal.gdalconst.GA_ReadOnly)
                tpi_data = tpi_src.ReadAsArray()

                ndvi_src = gdal.Open(resampled_modis)
                scale_factor = float(ndvi_src.GetMetadata()["scale_factor"])
                offset = float(ndvi_src.GetMetadata()["add_offset"])
                ndvi_data = ndvi_src.ReadAsArray()
                ndvi_data = (ndvi_data - offset) / scale_factor

                pred_X = pd.DataFrame(
                    columns=[
                        "S5PConcentration",
                        "Surface Solar Radiance",
                        "Surface Pressure",
                        "Total Precipitation",
                        "Topographic Position Index",
                        "NDVI",
                        "Day of Year",
                        "Total Seconds",
                        "Relative Humidity",
                        "Wind Speed",
                        "Wind Direction",
                        "Distance to Center",
                    ]
                )

                pred_X["S5PConcentration"] = pd.Series(list(co_tropomi_value.flatten()))
                pred_X["Topographic Position Index"] = pd.Series(list(tpi_data.flatten()))
                pred_X["Day of Year"] = int(request_datetime.strftime("%j"))
                pred_X["NDVI"] = pd.Series(list(ndvi_data.flatten()))

                # Create output file same as s5p
                output_path = os.path.join(RESULTS_DIR, request_datetime.strftime("%Y-%m-%d"))
                os.makedirs(
                    output_path,
                    exist_ok=True,
                )

                driver = gdal.GetDriverByName("GTiff")
                output = os.path.join(
                    output_path,
                    "SR_" + S5P_file.split(".")[0] + ".tiff",
                )
                out = driver.Create(output, s5p_src.RasterXSize, s5p_src.RasterYSize, 1, gdal.GDT_Float32)
                out.SetGeoTransform(s5p_src.GetGeoTransform())
                out.SetProjection(s5p_src.GetProjection())
                out.GetRasterBand(1).SetNoDataValue(9.96921e36)

                x2, y2 = get_time_index(lon_in, lat_in, (lon_bounds[0] + lon_bounds[1]) / 2,
                                        (lat_bounds[0] + lat_bounds[1]) / 2)
                t1 = pd.Timestamp(impact_delta[0, x2])

                timestamp_start = t1.normalize()  # or use timestamp_end.floor('D')
                # Calculate the difference as a timedelta
                time_difference = t1 - timestamp_start
                # Convert the timedelta to total seconds
                total_seconds = time_difference.total_seconds()

                pred_X["Total Seconds"] = total_seconds

                dt_index = int(pd.Timestamp(t1.value).hour)

                # Distance to center
                lon_center = (-25.0 + 45.0) / 2
                lat_center = (30.0 + 72.0) / 2

                # find all longitude and latitude resampled
                GT = s5p_src.GetGeoTransform()
                x_size = s5p_src.RasterXSize
                y_size = s5p_src.RasterYSize

                latitudes = GT[3] + np.arange(y_size) * GT[5]
                longitudes = GT[0] + np.arange(x_size) * GT[1]

                latitude_grid, longitude_grid = np.meshgrid(latitudes, longitudes, indexing="ij")

                distances = [
                    np.sqrt(haversine((lat, lon), (lat_center, lon_center)))
                    for lat, lon in zip(latitude_grid.flatten(), longitude_grid.flatten())
                ]

                pred_X["Distance to Center"] = distances

                d2m = np.zeros((s5p_src.RasterYSize, s5p_src.RasterXSize))
                t2m = np.zeros((s5p_src.RasterYSize, s5p_src.RasterXSize))
                ssr = np.zeros((s5p_src.RasterYSize, s5p_src.RasterXSize))
                u10 = np.zeros((s5p_src.RasterYSize, s5p_src.RasterXSize))
                v10 = np.zeros((s5p_src.RasterYSize, s5p_src.RasterXSize))
                sp = np.zeros((s5p_src.RasterYSize, s5p_src.RasterXSize))
                tp = np.zeros((s5p_src.RasterYSize, s5p_src.RasterXSize))

                for value in ["d2m", "t2m", "ssr", "u10", "v10", "sp", "tp"]:
                    era5_outpath = os.path.join(
                        RESAMPLING_DIR, f"Resampled_ERA5_Land_{request_datetime.strftime('%Y-%m-%d')}_{value}.tiff"
                    )
                    era5_src = gdal.Open(era5_outpath, gdal.gdalconst.GA_ReadOnly)

                    if value == "d2m":
                        band1 = era5_src.GetRasterBand(dt_index + 1)
                        value1 = band1.ReadAsArray()

                        band2 = era5_src.GetRasterBand(dt_index + 2)
                        value2 = band2.ReadAsArray()

                        d2m = (
                                      value2 * (float(pd.Timestamp(impact_delta[0, x2]).minute))
                                      + value1 * (60 - float(pd.Timestamp(impact_delta[0, x2]).minute))
                              ) / (60)

                    elif value == "t2m":
                        band1 = era5_src.GetRasterBand(dt_index + 1)
                        value1 = band1.ReadAsArray()

                        band2 = era5_src.GetRasterBand(dt_index + 2)
                        value2 = band2.ReadAsArray()

                        t2m = (
                                      value2 * (float(pd.Timestamp(impact_delta[0, x2]).minute))
                                      + value1 * (60 - float(pd.Timestamp(impact_delta[0, x2]).minute))
                              ) / (60)

                    elif value == "ssr":
                        band1 = era5_src.GetRasterBand(dt_index + 1)
                        value1 = band1.ReadAsArray()

                        band2 = era5_src.GetRasterBand(dt_index + 2)
                        value2 = band2.ReadAsArray()

                        ssr = (
                                      value2 * (float(pd.Timestamp(impact_delta[0, x2]).minute))
                                      + value1 * (60 - float(pd.Timestamp(impact_delta[0, x2]).minute))
                              ) / (60)

                    elif value == "u10":
                        band1 = era5_src.GetRasterBand(dt_index + 1)
                        value1 = band1.ReadAsArray()

                        band2 = era5_src.GetRasterBand(dt_index + 2)
                        value2 = band2.ReadAsArray()

                        u10 = (
                                      value2 * (float(pd.Timestamp(impact_delta[0, x2]).minute))
                                      + value1 * (60 - float(pd.Timestamp(impact_delta[0, x2]).minute))
                              ) / (60)

                    elif value == "v10":
                        band1 = era5_src.GetRasterBand(dt_index + 1)
                        value1 = band1.ReadAsArray()

                        band2 = era5_src.GetRasterBand(dt_index + 2)
                        value2 = band2.ReadAsArray()

                        v10 = (
                                      value2 * (float(pd.Timestamp(impact_delta[0, x2]).minute))
                                      + value1 * (60 - float(pd.Timestamp(impact_delta[0, x2]).minute))
                              ) / (60)

                    elif value == "sp":
                        band1 = era5_src.GetRasterBand(dt_index + 1)
                        value1 = band1.ReadAsArray()

                        band2 = era5_src.GetRasterBand(dt_index + 2)
                        value2 = band2.ReadAsArray()

                        sp = (
                                     value2 * (float(pd.Timestamp(impact_delta[0, x2]).minute))
                                     + value1 * (60 - float(pd.Timestamp(impact_delta[0, x2]).minute))
                             ) / (60)

                    elif value == "tp":
                        band1 = era5_src.GetRasterBand(dt_index + 1)
                        value1 = band1.ReadAsArray()

                        band2 = era5_src.GetRasterBand(dt_index + 2)
                        value2 = band2.ReadAsArray()

                        tp = (
                                     value2 * (float(pd.Timestamp(impact_delta[0, x2]).minute))
                                     + value1 * (60 - float(pd.Timestamp(impact_delta[0, x2]).minute))
                             ) / (60)

                pred_X["Relative Humidity"] = (6.1078 * np.exp((17.1 * d2m.flatten()) / (235 + d2m.flatten()))) / (
                        6.1078 * np.exp((17.1 * t2m.flatten()) / (235 + t2m.flatten()))
                )
                pred_X["Wind Speed"] = np.sqrt(pow(u10.flatten(), 2) + pow(v10.flatten(), 2))
                pred_X["Wind Direction"] = np.fmod(180 + (180 / math.pi) * np.arctan2(v10.flatten(), u10.flatten()),
                                                   360)
                pred_X["Surface Solar Radiance"] = pd.Series(list(ssr.flatten()))
                pred_X["Surface Pressure"] = pd.Series(list(sp.flatten()))
                pred_X["Total Precipitation"] = pd.Series(list(tp.flatten()))

                pred_X["Relative Humidity"] = pred_X["Relative Humidity"].fillna(-1)
                pred_X["Wind Speed"] = pred_X["Wind Speed"].fillna(-1)
                pred_X["Wind Direction"] = pred_X["Wind Direction"].fillna(-1)
                pred_X["Surface Solar Radiance"] = pred_X["Surface Solar Radiance"].fillna(-1)
                pred_X["Surface Pressure"] = pred_X["Surface Pressure"].fillna(-1)
                pred_X["Total Precipitation"] = pred_X["Total Precipitation"].fillna(-1)
                pred_X["S5PConcentration"] = pred_X["S5PConcentration"].fillna(-1)

                not_scale_cols = ["Day of Year", "Total Seconds"]  # Ground CO Concentration is the target
                scale_cols = pred_X.drop(columns=not_scale_cols).columns

                # Apply sine transformation to cyclic features
                for col, max_val in zip(["Day of Year", "Total Seconds"], [365, 86400]):
                    pred_X[col] = np.sin(2 * np.pi * pred_X[col] / max_val)

                # Initialize and apply StandardScaler
                pred_X_scaled = pd.DataFrame(scaler.transform(pred_X[scale_cols]), columns=scale_cols)
                normalized = pd.concat([pred_X_scaled, pred_X[not_scale_cols]], axis=1)

                data = normalized.values.tolist()
                import json
                response = requests.post(url="http://cm-ray-service-head-svc.ray:8000/predict",
                                         data=json.dumps({"data": data}))

                predictions_list = response.json()["predictions"]
                predictions = pd.DataFrame(predictions_list, columns=["Ground CO Concentration"])

                predictions[np.isnan(co_tropomi_value.flatten())] = 9.96921e36
                predictions[co_tropomi_value.flatten() < 0] = 9.96921e36
                predictions[co_tropomi_value.flatten() == 9.96921e36] = 9.96921e36
                predictions[np.isnan(tp.flatten())] = 9.96921e36
                predictions[predictions < 0] = 0
                predictions[pred_X["NDVI"] > 1] = 9.96921e36
                predictions[pred_X["NDVI"] < -1] = 9.96921e36

                out.WriteArray(np.reshape(predictions, (s5p_src.RasterYSize, s5p_src.RasterXSize)))
                out.FlushCache()

                predictions = None

            era5_src = None
            s5p_src = None
            band = None
            co_tropomi_value = None
            out = None
            pred_X = None
            u10 = None
            v10 = None
            d2m = None
            t2m = None
            ssr = None
            sp = None
            tp = None

        ndvi_src = None
        ndvi_data = None

    predict(date)


carbon_monoxide_production_dag(date="20240206")
