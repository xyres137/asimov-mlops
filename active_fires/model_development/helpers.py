def generate_random_values(n, seed=None):
    import random

    if seed is not None:
        random.seed(seed)
    return [random.randint(0, 100) for _ in range(n)]


def load_samples(csv_file, rel_path, class_id=2):
    import os
    import pandas as pd

    data = pd.read_csv(csv_file, index_col=False)

    if class_id == 0 or class_id == 1 or class_id == -1:
        data = data[data["Class_2"] == class_id]

    data = data.sample(frac=1)
    data = data[
        [
            "Α/Α ΕΓΓΡΑΦΗΣ",
            "X-ENGAGE",
            "Y-ENGAGE",
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
            "MODIS_RAD_NODATA_PIXELS",
            "MODIS_FIRE_PIXELS",
            "Class",
            "Class_2",
        ]
    ]

    modis_rad = list(data.iloc[:, 3])
    seviri = list(data.iloc[:, 5])
    seviri_hrv = list(data.iloc[:, 6])
    elevation = list(data.iloc[:, 7])
    slope = list(data.iloc[:, 8])
    aspect = list(data.iloc[:, 9])
    tpi = list(data.iloc[:, 10])
    fuel = list(data.iloc[:, 11])
    height = list(data.iloc[:, 12])
    cover = list(data.iloc[:, 13])
    class_id = list(data.iloc[:, 17])

    samples = []
    for (
        modis_rad,
        seviri,
        seviri_hrv,
        elevation,
        slope,
        aspect,
        tpi,
        fuel,
        height,
        cover,
        class_id,
    ) in zip(
        modis_rad,
        seviri,
        seviri_hrv,
        elevation,
        slope,
        aspect,
        tpi,
        fuel,
        height,
        cover,
        class_id,
    ):
        samples.append(
            [
                os.path.join(rel_path, modis_rad),
                os.path.join(rel_path, seviri),
                os.path.join(rel_path, seviri_hrv),
                os.path.join(rel_path, elevation),
                os.path.join(rel_path, slope),
                os.path.join(rel_path, aspect),
                os.path.join(rel_path, tpi),
                os.path.join(rel_path, fuel),
                os.path.join(rel_path, height),
                os.path.join(rel_path, cover),
                class_id,
                "none",
            ]
        )

    return samples


def data_generator(samples, batch_size, input_features):
    import os
    import numpy as np
    from osgeo import gdal
    import tensorflow as tf

    num_samples = len(samples)

    while True:
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset + batch_size]

            seviri = []
            modis = []
            elevation_set = []
            slope_set = []
            aspect_set = []
            tpi_set = []
            fuel_set = []
            cover_set = []
            height_set = []

            for batch_sample in batch_samples:
                labeled = batch_sample[0]  # MODIS Fire labels

                input_name_seviri = batch_sample[1]
                input_name_seviri_hrv = batch_sample[2]
                elevation = batch_sample[3]
                slope = batch_sample[4]
                aspect = batch_sample[5]
                tpi = batch_sample[6]
                fuel = batch_sample[7]
                height = batch_sample[8]
                cover = batch_sample[9]

                label = gdal.Open(labeled)
                labeled_data = label.GetRasterBand(4).ReadAsArray()
                labeled_data[labeled_data <= 7] = 0
                labeled_data[labeled_data == 32767] = 0
                labeled_data[labeled_data > 7] = 1
                labeled_data[labeled_data > 9] = 0
                labeled_data[labeled_data < 0] = 0
                index = np.isnan(labeled_data)
                labeled_data[index] = 0

                input_seviri = np.array([])
                input_name_seviri_ = gdal.Open(input_name_seviri)

                if "seviri_vis06_" in input_features:
                    input_name_seviri_data_b1 = input_name_seviri_.GetRasterBand(1).ReadAsArray()
                    input_seviri = np.append(input_seviri, input_name_seviri_data_b1 / 1000.0)
                if "seviri_vis08_" in input_features:
                    input_name_seviri_data_b2 = input_name_seviri_.GetRasterBand(2).ReadAsArray()
                    input_seviri = np.append(input_seviri, input_name_seviri_data_b2 / 1000.0)
                if "seviri_nir16_" in input_features:
                    input_name_seviri_data_b3 = input_name_seviri_.GetRasterBand(3).ReadAsArray()
                    input_seviri = np.append(input_seviri, input_name_seviri_data_b3 / 1000.0)
                if "seviri_ir39_" in input_features:
                    input_name_seviri_data_b4 = input_name_seviri_.GetRasterBand(4).ReadAsArray()
                    input_seviri = np.append(input_seviri, input_name_seviri_data_b4 / 1000.0)
                if "seviri_wv62_" in input_features:
                    input_name_seviri_data_b5 = input_name_seviri_.GetRasterBand(5).ReadAsArray()
                    input_seviri = np.append(input_seviri, input_name_seviri_data_b5 / 1000.0)
                if "seviri_wv73_" in input_features:
                    input_name_seviri_data_b6 = input_name_seviri_.GetRasterBand(6).ReadAsArray()
                    input_seviri = np.append(input_seviri, input_name_seviri_data_b6 / 1000.0)
                if "seviri_ir87_" in input_features:
                    input_name_seviri_data_b7 = input_name_seviri_.GetRasterBand(7).ReadAsArray()
                    input_seviri = np.append(input_seviri, input_name_seviri_data_b7 / 1000.0)
                if "seviri_ir97_" in input_features:
                    input_name_seviri_data_b8 = input_name_seviri_.GetRasterBand(8).ReadAsArray()
                    input_seviri = np.append(input_seviri, input_name_seviri_data_b8 / 1000.0)
                if "seviri_ir108_" in input_features:
                    input_name_seviri_data_b9 = input_name_seviri_.GetRasterBand(9).ReadAsArray()
                    input_seviri = np.append(input_seviri, input_name_seviri_data_b9 / 1000.0)
                if "seviri_ir12_" in input_features:
                    input_name_seviri_data_b10 = input_name_seviri_.GetRasterBand(10).ReadAsArray()
                    input_seviri = np.append(input_seviri, input_name_seviri_data_b10 / 1000.0)
                if "seviri_ir134_" in input_features:
                    input_name_seviri_data_b11 = input_name_seviri_.GetRasterBand(11).ReadAsArray()
                    input_seviri = np.append(input_seviri, input_name_seviri_data_b11 / 1000.0)

                if "seviri_hrv" in input_features:
                    input_name_seviri_hrv_ = gdal.Open(input_name_seviri_hrv)
                    input_name_seviri_hrv_data = input_name_seviri_hrv_.ReadAsArray()
                    input_seviri = np.append(input_seviri, input_name_seviri_hrv_data / 255.0)

                if "seviri_ir39/seviri_nir16" in input_features:
                    input_name_seviri_data_b4 = input_name_seviri_.GetRasterBand(4).ReadAsArray() / 1000.0
                    input_name_seviri_data_b3 = input_name_seviri_.GetRasterBand(3).ReadAsArray() / 1000.0
                    input_name_seviri_data_b3[input_name_seviri_data_b3 == 0] = 0.001
                    input_seviri = np.append(
                        input_seviri,
                        input_name_seviri_data_b4 / input_name_seviri_data_b3,
                    )

                if "seviri_ir108/seviri_ir39" in input_features:
                    input_name_seviri_data_b9 = input_name_seviri_.GetRasterBand(9).ReadAsArray() / 1000.0
                    input_name_seviri_data_b4 = input_name_seviri_.GetRasterBand(4).ReadAsArray() / 1000.0
                    input_name_seviri_data_b4[input_name_seviri_data_b4 == 0] = 0.001
                    input_seviri = np.append(
                        input_seviri,
                        input_name_seviri_data_b9 / input_name_seviri_data_b4,
                    )

                if "seviri_ir39-seviri_ir108" in input_features:
                    input_name_seviri_data_b9 = input_name_seviri_.GetRasterBand(9).ReadAsArray() / 1000.0
                    input_name_seviri_data_b4 = input_name_seviri_.GetRasterBand(4).ReadAsArray() / 1000.0
                    input_seviri = np.append(
                        input_seviri,
                        input_name_seviri_data_b4 - input_name_seviri_data_b9,
                    )

                if "zscore-seviri_vis06" in input_features:
                    input_name_seviri_data_b1 = input_name_seviri_.GetRasterBand(1).ReadAsArray()
                    input_seviri = np.append(
                        input_seviri,
                        tf.image.per_image_standardization(input_name_seviri_data_b1[None, :, :]),
                    )

                if "zscore-seviri_nir16" in input_features:
                    input_name_seviri_data_b3 = input_name_seviri_.GetRasterBand(3).ReadAsArray()
                    input_seviri = np.append(
                        input_seviri,
                        tf.image.per_image_standardization(input_name_seviri_data_b3[None, :, :]),
                    )

                if "zscore-seviri_ir39" in input_features:
                    input_name_seviri_data_b4 = input_name_seviri_.GetRasterBand(4).ReadAsArray()
                    input_seviri = np.append(
                        input_seviri,
                        tf.image.per_image_standardization(input_name_seviri_data_b4[None, :, :]),
                    )

                if "zscore-seviri_ir108" in input_features:
                    input_name_seviri_data_b9 = input_name_seviri_.GetRasterBand(9).ReadAsArray()
                    input_seviri = np.append(
                        input_seviri,
                        tf.image.per_image_standardization(input_name_seviri_data_b9[None, :, :]),
                    )

                if "zscore-seviri_ir12" in input_features:
                    input_name_seviri_data_b10 = input_name_seviri_.GetRasterBand(10).ReadAsArray()
                    input_seviri = np.append(
                        input_seviri,
                        tf.image.per_image_standardization(input_name_seviri_data_b10[None, :, :]),
                    )

                if "time" in input_features:
                    timestamp = os.path.split(labeled)[1].split(".")[2]
                    h = timestamp.zfill(4)[0:2]
                    m = timestamp.zfill(4)[2:]
                    seconds = int(h) * 3600 + int(m) * 60 + 60
                    time_feature = np.full(labeled_data.shape, np.sin(2 * np.pi * seconds / 86400.0))
                    input_seviri = np.append(input_seviri, time_feature)

                if "elevation" in input_features:
                    elevation_ds = gdal.Open(elevation)
                    elevation_data = elevation_ds.ReadAsArray()
                    elevation_data[elevation_data == -9999] = 0
                    input_seviri = np.append(input_seviri, elevation_data / 2977.0)

                if "slope" in input_features:
                    slope_ds = gdal.Open(slope)
                    slope_data = slope_ds.ReadAsArray()
                    slope_data[np.isnan(slope_data)] = 0
                    input_seviri = np.append(input_seviri, slope_data / 40.0)

                if "aspect" in input_features:
                    aspect_ds = gdal.Open(aspect)
                    aspect_data = aspect_ds.ReadAsArray()
                    aspect_data[aspect_data == -32768] = 0
                    aspect_data = np.sin(2 * np.pi * aspect_data / 360.0)
                    input_seviri = np.append(input_seviri, aspect_data)

                if "tpi" in input_features:
                    tpi_ds = gdal.Open(tpi)
                    tpi_data = tpi_ds.ReadAsArray()
                    tpi_data[tpi_data == -99999] = 0
                    tpi_data = (tpi_data + 18) / (14 + 18)  # -17.45 to 13.919
                    input_seviri = np.append(input_seviri, tpi_data)

                if "fuel" in input_features:
                    fuel_ds = gdal.Open(fuel)
                    fuel_data = fuel_ds.ReadAsArray()
                    fuel_data[np.isnan(fuel_data)] = 0
                    input_seviri = np.append(input_seviri, fuel_data / 10.0)

                if "cover" in input_features:
                    cover_ds = gdal.Open(cover)
                    cover_data = cover_ds.ReadAsArray()
                    cover_data[cover_data == -32768] = 0
                    input_seviri = np.append(input_seviri, cover_data)

                if "height" in input_features:
                    height_ds = gdal.Open(height)
                    height_data = height_ds.ReadAsArray()
                    height_data[height_data == 3.4e38] = 0
                    input_seviri = np.append(input_seviri, height_data / 33.0)

                input_seviri = np.reshape(
                    input_seviri,
                    (
                        len(input_features),
                        input_name_seviri_.RasterYSize,
                        input_name_seviri_.RasterXSize,
                    ),
                )

                seviri.append(input_seviri)
                modis.append(labeled_data[None, :, :])

            seviri = np.array(seviri)
            modis = np.array(modis)

            seviri = seviri.transpose((0, 2, 3, 1))
            modis = modis.transpose((0, 2, 3, 1))

            yield seviri, modis


def custom_binary_crossentropy(y_true, y_pred):
    import tensorflow as tf
    import tensorflow.keras.backend as K

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    bin_crossentropy = K.binary_crossentropy(y_true, y_pred)
    return K.mean(bin_crossentropy)


def dice_coeff(y_true, y_pred):
    import tensorflow as tf
    import tensorflow.keras.backend as K

    smooth = 1.0

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    score = (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    return 1 - dice_coeff(y_true, y_pred)


def recall_m(y_true, y_pred):
    import tensorflow as tf
    import tensorflow.keras.backend as K

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    import tensorflow as tf
    import tensorflow.keras.backend as K

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    import tensorflow as tf
    import tensorflow.keras.backend as K

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def autoencoder(input_shape=(128, 128, 1), filters=64, activation="relu", final_activation="sigmoid"):
    from tensorflow.keras import models
    from tensorflow.keras.layers import (
        Input,
        Conv2D,
        MaxPool2D,
        Conv2DTranspose,
        BatchNormalization,
        UpSampling2D,
    )

    input = Input(shape=input_shape)

    x = Conv2D(filters, (3, 3), (1, 1), activation=activation, padding="same")(input)
    x = BatchNormalization()(x)

    x = Conv2D(filters, (3, 3), (2, 2), activation=activation, padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(strides=(1, 1), padding="same")(x)

    x = Conv2D(filters * 2, (3, 3), (1, 1), activation=activation, padding="same")(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters * 2, (3, 3), (2, 2), activation=activation, padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(strides=(1, 1), padding="same")(x)

    x = Conv2D(filters * 4, (3, 3), (1, 1), activation=activation, padding="same")(x)
    x = BatchNormalization()(x)

    # Decoder
    x = Conv2DTranspose(filters * 4, 1, 1, activation=activation, padding="same")(x)
    x = Conv2D(filters * 4, (3, 3), activation=activation, padding="same")(x)
    x = Conv2D(filters * 2, (3, 3), activation=activation, padding="same")(x)

    x = Conv2DTranspose(filters * 2, 1, 1, padding="same")(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters * 2, (3, 3), activation=activation, padding="same")(x)
    x = Conv2D(filters, (3, 3), activation=activation, padding="same")(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters, (3, 3), padding="same")(x)

    output = Conv2D(1, 1, activation=final_activation)(x)

    model = models.Model(input, output, name="autoencoder")
    return model
