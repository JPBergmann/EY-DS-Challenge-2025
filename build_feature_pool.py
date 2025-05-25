from feature_engineering import (
    calculate_airquality_features,
    calculate_building_features,
    calculate_cooltower_features,
    calculate_emissions_features,
    calculate_landsat_satellite_features,
    calculate_sentinel_satellite_features,
    calculate_traffic_features,
    calculate_weather_features,
)

import pandas as pd
import numpy as np
import gc

def clean_col_names(dataframe):
    df = dataframe.copy()
    df.columns = [col.replace("[", " ") for col in df.columns]
    df.columns = [col.replace("]", " ") for col in df.columns]
    df.columns = [col.replace(",", " ") for col in df.columns]
    df.columns = [col.replace(" ", "_") for col in df.columns]
    return df

def drop_nan_over(dataframe, threshold):
    df = dataframe.copy()
    drop_cols = df.columns[(df.isna().sum() / len(df) > threshold)]
    df = df.drop(columns=drop_cols)
    return df

def main():
    train = pd.read_csv("./DATA/DATA_CLEAN/train.csv")
    test = pd.read_csv("./DATA/DATA_CLEAN/test.csv")

    AGG_FUNCS = ["mean", "min", "max", "sum", "std", "median", "skew"]
    BUFFER_DISTS_SAT = [5, 10, 25, 50, 100, 150, 200, 300]
    BUFFER_DISTS_BUILD = [25, 50, 75, 100, 250, 500, 750, 1000]


    # Train DATA

    # FIX DUPLICATES
    train_dfs = []

    sentinel = calculate_sentinel_satellite_features(train, "./DATA/DATA_CLEAN/Sentinel2_Median.tiff", BUFFER_DISTS_SAT, AGG_FUNCS)
    train_dfs.append(sentinel.drop(columns=["Latitude", "Longitude", "datetime"]))

    landsat = calculate_landsat_satellite_features(train, "./DATA/DATA_CLEAN/Landsat8_Median.tiff", BUFFER_DISTS_SAT, AGG_FUNCS)
    train_dfs.append(landsat.drop(columns=["Latitude", "Longitude", "datetime", "UHI Index"]))

    buildings = calculate_building_features(train, "./DATA/DATA_CLEAN/ny_buildings.geojson", BUFFER_DISTS_BUILD, AGG_FUNCS)
    train_dfs.append(buildings.drop(columns=["Latitude", "Longitude", "datetime", "UHI Index"]))

    cooltowers = calculate_cooltower_features(train, "./DATA/DATA_CLEAN/ny_cooltowers.geojson", BUFFER_DISTS_BUILD, AGG_FUNCS)
    train_dfs.append(cooltowers.drop(columns=["Latitude", "Longitude", "datetime", "UHI Index"]))

    emissions = calculate_emissions_features(train, "./DATA/DATA_CLEAN/ny_emissions.geojson", BUFFER_DISTS_BUILD, AGG_FUNCS)
    train_dfs.append(emissions.drop(columns=["Latitude", "Longitude", "datetime", "UHI Index"]))

    traffic = calculate_traffic_features(train, "./DATA/DATA_CLEAN/ny_traffic.geojson", BUFFER_DISTS_BUILD, AGG_FUNCS)
    train_dfs.append(traffic.drop(columns=["Latitude", "Longitude", "datetime", "UHI Index"]))

    airquality = calculate_airquality_features(train, "./DATA/DATA_CLEAN/ny_airquality.geojson")
    train_dfs.append(airquality.drop(columns=["Latitude", "Longitude", "datetime", "UHI Index"]))

    # weather = calculate_weather_features(train, "./DATA/DATA_CLEAN/ny_weather.xlsx", AGG_FUNCS)
    # train_dfs.append(weather.drop(columns=["Latitude", "Longitude", "datetime", "UHI Index"]))

    train = pd.concat(train_dfs, axis=1)
    print(f"Train shape pre-NAN drop: {train.shape}")
    train = train.replace([np.inf, -np.inf], np.nan)
    train = clean_col_names(train)
    train = drop_nan_over(train, 0.5)
    print(f"Train shape post-NAN drop: {train.shape}")

    DATA_COLS = train.columns

    print(f"Writing train data to parquet file -> Shape: {train.shape}")
    train.to_parquet("./DATA/DATA_CLEAN/all_train_features.parquet")

    # Free up memory
    del train_dfs
    gc.collect()

    # Test DATA
    test_dfs = []

    sentinel = calculate_sentinel_satellite_features(test, "./DATA/DATA_CLEAN/Sentinel2_Median.tiff", BUFFER_DISTS_SAT, AGG_FUNCS)
    test_dfs.append(sentinel.drop(columns=["Latitude", "Longitude", "UHI Index"]))

    landsat = calculate_landsat_satellite_features(test, "./DATA/DATA_CLEAN/Landsat8_Median.tiff", BUFFER_DISTS_SAT, AGG_FUNCS)
    test_dfs.append(landsat.drop(columns=["Latitude", "Longitude", "UHI Index"]))

    buildings = calculate_building_features(test, "./DATA/DATA_CLEAN/ny_buildings.geojson", BUFFER_DISTS_BUILD, AGG_FUNCS)
    test_dfs.append(buildings.drop(columns=["Latitude", "Longitude", "UHI Index"]))

    cooltowers = calculate_cooltower_features(test, "./DATA/DATA_CLEAN/ny_cooltowers.geojson", BUFFER_DISTS_BUILD, AGG_FUNCS)
    test_dfs.append(cooltowers.drop(columns=["Latitude", "Longitude", "UHI Index"]))

    emissions = calculate_emissions_features(test, "./DATA/DATA_CLEAN/ny_emissions.geojson", BUFFER_DISTS_BUILD, AGG_FUNCS)
    test_dfs.append(emissions.drop(columns=["Latitude", "Longitude", "UHI Index"]))

    traffic = calculate_traffic_features(test, "./DATA/DATA_CLEAN/ny_traffic.geojson", BUFFER_DISTS_BUILD, AGG_FUNCS)
    test_dfs.append(traffic.drop(columns=["Latitude", "Longitude", "UHI Index"]))

    airquality = calculate_airquality_features(test, "./DATA/DATA_CLEAN/ny_airquality.geojson")
    test_dfs.append(airquality.drop(columns=["Latitude", "Longitude", "UHI Index"]))

    # weather = calculate_weather_features(test, "./DATA/DATA_CLEAN/ny_weather.xlsx", AGG_FUNCS)
    # test_dfs.append(weather.drop(columns=["Latitude", "Longitude", "UHI Index"]))

    test = pd.concat(test_dfs, axis=1)
    print(f"Test shape pre-NAN drop: {test.shape}")
    test = test.replace([np.inf, -np.inf], np.nan)
    test = clean_col_names(test)
    test = test[DATA_COLS.drop("UHI_Index")]
    print(f"Test shape post-NAN drop: {test.shape}")

    print(f"Writing test data to parquet file -> Shape: {test.shape}")
    test.to_parquet("./DATA/DATA_CLEAN/all_test_features.parquet")

    # Free up memory
    del test_dfs
    gc.collect()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)