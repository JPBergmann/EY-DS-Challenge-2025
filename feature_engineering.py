# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

# UI/Plotting
from tqdm.auto import tqdm

# IO/Data wrangling
import gc
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rio
import scipy.stats as stats

def calculate_cooltower_features(data_df, geojson_path, buffer_distances, agg_levels):
    data = data_df.copy()

    # Progress bar
    progbar = tqdm(total=len(buffer_distances), desc="Reading Cooling Tower data")

    # Read cooling tower data
    cooling_towers = gpd.read_file(geojson_path, crs=4326)
    cooling_towers = cooling_towers[["geometry", "Active_Equip"]]

    # Prepare train data as GeoDataFrame
    data_geometry = gpd.GeoDataFrame(
        data[["Longitude", "Latitude"]], geometry=gpd.points_from_xy(data["Longitude"], data["Latitude"]), crs=4326
    )

    # Estimate UTM CRS from data points and reproject both GeoDataFrames to this CRS
    utm_crs = data_geometry.estimate_utm_crs()
    data_geometry = data_geometry.to_crs(utm_crs)
    cooling_towers = cooling_towers.to_crs(utm_crs)

    # Aggregation functions
    agg_funcs = {
        "Active_Equip": agg_levels,
    }

    # Iterate through buffer distances
    aggregations = []
    for distance in buffer_distances:
        # Create buffers for all points
        progbar.set_description(f"Calculating Cooling Tower Features for buffer {distance}m")
        data_geometry["buffer"] = data_geometry.geometry.buffer(distance)

        # Perform spatial join: cooling_towers within buffers
        buffer_gdf = data_geometry.set_geometry("buffer")
        joined = gpd.sjoin(cooling_towers, buffer_gdf, how="left", predicate="intersects")
        # Aggregate statistics for all features
        grouped = joined.groupby("index_right", dropna=False)
        for feature, funcs in agg_funcs.items():
            agg = grouped[feature].agg(funcs)
            agg.columns = [f"{func}_{feature}_{distance}m" for func in funcs]
            agg[f"num_cooltowers_{distance}m"] = grouped["index_right"].value_counts()
            aggregations.append(agg)

        # Update progress bar
        progbar.update(1)

    for agg in aggregations:
        data = data.merge(agg, left_index=True, right_index=True, how="left")

    progbar.close()

    return data

def calculate_airquality_features(data_df, geojson_path):
    data = data_df.copy()

    # Read airquality data
    air_quality = gpd.read_file(geojson_path, crs=4326)

    # Prepare train data as GeoDataFrame
    data_geometry = gpd.GeoDataFrame(
        data, geometry=gpd.points_from_xy(data["Longitude"], data["Latitude"]), crs=4326
    )

    # Spacial join
    data = gpd.sjoin(data_geometry, air_quality, how="left", predicate="intersects")

    return data.drop(columns=["index_right", "geometry"])

def calculate_building_features(data_df, geojson_path, buffer_distances, agg_levels):
    data = data_df.copy()

    # Progress bar
    progbar = tqdm(total=len(buffer_distances), desc="Reading Building Footprint data")

    # Read building footprints
    buildings = gpd.read_file(geojson_path, crs=4326)
    buildings = buildings.astype(
        {
            "heightroof": float,
            "cnstrct_yr": float,
            "groundelev": float,
            "feat_code": float,
        }
    )

    # Prepare train data as GeoDataFrame
    data_geometry = gpd.GeoDataFrame(
        data[["Longitude", "Latitude"]], geometry=gpd.points_from_xy(data["Longitude"], data["Latitude"]), crs=4326
    )

    # Estimate UTM CRS from data points and reproject both GeoDataFrames to this CRS
    utm_crs = data_geometry.estimate_utm_crs()
    data_geometry = data_geometry.to_crs(utm_crs)
    buildings = buildings.to_crs(utm_crs)

    # Aggregation functions
    # dens_cols = buildings.filter(regex="Density_bw_", axis=1).columns.to_list()
    #dens_cols = ["Density_bw_scott"]
    #agg_funcs = {col: ["mean"] for col in dens_cols}

    # cluster_cols = buildings.filter(regex="Cluster_", axis=1).columns.to_list()
    #cluster_cols = ["Cluster_mcs_5_ms_10"]
    #agg_funcs.update({col: [lambda x: stats.mode(x)[0]] for col in cluster_cols}) # Ugly and slow workaround for mode :/

    agg_funcs = {
        "heightroof": agg_levels,
        "cnstrct_yr": agg_levels,
        "groundelev": agg_levels,
    }

    # Iterate through buffer distances
    aggregations = []
    for distance in buffer_distances:
        # Create buffers for all points
        progbar.set_description(f"Calculating Building Features for buffer {distance}m")
        data_geometry["buffer"] = data_geometry.geometry.buffer(distance)

        # Perform spatial join: buildings within buffers
        buffer_gdf = data_geometry.set_geometry("buffer")
        joined = gpd.sjoin(buildings, buffer_gdf, how="left", predicate="intersects")

        # Aggregate statistics for all features
        grouped = joined.groupby("index_right", dropna=False)
        for feature, funcs in agg_funcs.items():
            # if feature in cluster_cols:
            #     agg = grouped[feature].agg(funcs)
            #     agg.columns = [f"mode_{feature}_{distance}m"]
            # else:
            agg = grouped[feature].agg(funcs)
            agg.columns = [f"{func}_{feature}_{distance}m" for func in funcs]
            aggregations.append(agg)

        # Count number of buildings by type
        counts = (
            grouped["feat_code"]
            .value_counts()
            .unstack(fill_value=0)
            .rename(
                columns={
                    1000: f"num_parking_{distance}m",
                    1001: f"num_gas_station_{distance}m",
                    1002: f"num_storage_tank_{distance}m",
                    1003: f"num_placeholder_{distance}m",
                    1004: f"num_auxiliary_{distance}m",
                    1005: f"num_temporary_{distance}m",
                    1006: f"num_cantilevered_{distance}m",
                    2100: f"num_building_{distance}m",
                    2110: f"num_skybridge_{distance}m",
                    5100: f"num_building_under_construction_{distance}m",
                    5110: f"num_garage_{distance}m",
                }
            )
        )
        counts[f"num_total_structures_{distance}m"] = counts.sum(axis=1)
        aggregations.append(counts)

        # Update progress bar
        progbar.update(1)

    for agg in aggregations:
        data = data.merge(agg, left_index=True, right_index=True, how="left")

    progbar.close()

    return data

def calculate_landsat_satellite_features(data_df, tiff_path, buffer_distances, agg_levels):
    data = data_df.copy()

    # Progress bar
    progbar = tqdm(total=len(buffer_distances), desc="Reading Landsat8 Satellite data")

    # Load raster lazily with Dask
    landsat_raster = rio.open_rasterio(tiff_path, chunks="auto")

    # Prepare train data as GeoDataFrame
    data_geometry = gpd.GeoDataFrame(data[["Longitude", "Latitude"]], geometry=gpd.points_from_xy(data["Longitude"], data["Latitude"]), crs=4326)

    # Estimate UTM CRS from data points and reproject both datasets to this CRS
    utm_crs = data_geometry.estimate_utm_crs()
    data_geometry = data_geometry.to_crs(utm_crs)
    landsat_raster = landsat_raster.rio.reproject(utm_crs)

    # Aggregation functions
    band_names = ["coastal", "blue", "green", "red", "nir08", "swir16", "swir22", "lwir11"]
    # agg_funcs = {band: agg_levels for band in band_names}
    agg_funcs = {}
    agg_funcs.update({
        "LST": agg_levels,
        "CRI": agg_levels,
        "TUI": agg_levels,
        "BAI": agg_levels,
        "SUI": agg_levels,
        "BAEI": agg_levels,
        "UTFVI": agg_levels,
        "LSE": agg_levels,
        "UHLI": agg_levels,
        "STVI": agg_levels,
        "ASI": agg_levels,
        "EBBI": agg_levels,
        "TVI": agg_levels,
        "TCI": agg_levels,
        "VHI": agg_levels,
        "Albedo": agg_levels,
    })

    # Iterate through buffer distances
    aggregations = []
    for distance in buffer_distances:
        # Create buffers for all points
        progbar.set_description(f"Calculating Landsat8 Satellite Features for buffer {distance}m")
        data_geometry["buffer"] = data_geometry.geometry.buffer(distance)

        # Calculate bounding box of buffered points
        minx, miny, maxx, maxy = data_geometry["buffer"].total_bounds

        # Subset raster by bounding box
        subset_raster = landsat_raster.rio.clip_box(minx, miny, maxx, maxy)

        # Convert subset raster to DataFrame
        band_coords = subset_raster.to_dataframe(name="pixel_value").reset_index()
        band_coords = band_coords.pivot_table(index=["y", "x"], columns="band", values="pixel_value").reset_index()
        band_coords.columns.name = None
        band_coords.columns = ["y", "x"] + band_names
        band_coords = gpd.GeoDataFrame(band_coords, geometry=gpd.points_from_xy(band_coords["x"], band_coords["y"]), crs=utm_crs)

        # Reduce float precision
        band_coords[band_names] = band_coords[band_names].astype(np.float32)

        # Calculate spectral indices
        ndvi = (band_coords["nir08"] - band_coords["red"]) / (band_coords["nir08"] + band_coords["red"])
        band_coords["LST"] = band_coords["lwir11"]
        band_coords["CI"] = band_coords["LST"] / ndvi
        band_coords["CRI"] = (band_coords["swir22"] / band_coords["swir16"])
        band_coords["TUI"] = (band_coords["lwir11"] / ndvi)
        band_coords["BAI"] = (1.0 / ((0.1 - band_coords["red"]) ** 2 + (0.06 - band_coords["nir08"]) ** 2))
        band_coords["SUI"] = (band_coords["swir16"] + band_coords["nir08"]) / (band_coords["blue"] + band_coords["green"])
        band_coords["BAEI"] = (band_coords["red"] + band_coords["swir16"]) / band_coords["nir08"]
        band_coords["UTFVI"] = (band_coords["lwir11"] - ndvi) / ndvi
        band_coords["LSE"] = 0.004 * ndvi + 0.986
        band_coords["UHLI"] = band_coords["lwir11"] * (1 - ndvi)
        band_coords["STVI"] = (band_coords["lwir11"] - ndvi) / (band_coords["lwir11"] + ndvi)
        band_coords["ASI"] = (band_coords["swir16"] + band_coords["swir22"]) / (band_coords["nir08"] + band_coords["red"])
        band_coords["EBBI"] = (band_coords["swir22"] - band_coords["nir08"]) / (band_coords["swir22"] + 10 * (band_coords["red"] ** 0.5))
        band_coords["TVI"] = np.sqrt((band_coords["nir08"] - band_coords["red"]) / (band_coords["nir08"] + band_coords["red"]) + 0.5)
        LST_min = band_coords["lwir11"].min()
        LST_max = band_coords["lwir11"].max()
        band_coords["TCI"] = (band_coords["lwir11"] - LST_min) / (LST_max - LST_min)
        band_coords["VHI"] = 0.5 * (ndvi + band_coords["TCI"])
        coeffs = {
            'blue': 0.254,  # Blue band (B2)
            'green': 0.303, # Green band (B3)
            'red': 0.328,   # Red band (B4)
            'nir08': 0.132, # Near-infrared (B5)
            'swir16': 0.034, # SWIR (B6)
            'swir22': 0.012  # SWIR (B7)
        }
        band_coords["Albedo"] = np.sum([coeffs[band] * band_coords[band] for band in coeffs], axis=0)

        # Perform spatial join: pixels within buffers
        buffer_gdf = data_geometry.set_geometry("buffer")
        joined = gpd.sjoin(band_coords, buffer_gdf, how="left", predicate="intersects")

        # Aggregate statistics for all features
        grouped = joined.groupby("index_right")
        for feature, funcs in agg_funcs.items():
            agg = grouped[feature].agg(funcs)
            agg.columns = [f"{func}_{feature}_{distance}m" for func in funcs]
            aggregations.append(agg)

        # Update progress bar
        progbar.update(1)

        # Collect garbage
        gc.collect()

    # Merge all aggregations back to the original dataframe
    for agg in aggregations:
        data = pd.merge(data, agg, left_index=True, right_index=True, how="left")

    progbar.close()
    
    return data

def calculate_sentinel_satellite_features(data_df, tiff_path, buffer_distances, agg_levels):
    data = data_df.copy()

    # Progress bar
    progbar = tqdm(total=len(buffer_distances), desc="Reading Sentinel2 Satellite data")

    # Load raster lazily with Dask
    sentinel_raster = rio.open_rasterio(tiff_path, chunks="auto")

    # Prepare train data as GeoDataFrame
    data_geometry = gpd.GeoDataFrame(data[["Longitude", "Latitude"]], geometry=gpd.points_from_xy(data["Longitude"], data["Latitude"]), crs=4326)

    # Estimate UTM CRS from data points and reproject both datasets to this CRS
    utm_crs = data_geometry.estimate_utm_crs()
    data_geometry = data_geometry.to_crs(utm_crs)
    sentinel_raster = sentinel_raster.rio.reproject(utm_crs)

    # Aggregation functions
    band_names = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    # agg_funcs = {band: agg_levels for band in band_names}
    agg_funcs = {}
    agg_funcs.update({
        "NDVI": agg_levels,
        "BSI": agg_levels,
        "BSI2": agg_levels,
        "NDBI": agg_levels,
        "NBSI": agg_levels,
        "UI": agg_levels,
        "NDWI": agg_levels,
        "AWEI": agg_levels,
        "MNDWI": agg_levels,
        "SAVI": agg_levels,
        "EVI": agg_levels,
        "NDBal": agg_levels,
        "ISA": agg_levels,
        "NDISI": agg_levels,
        "NBI": agg_levels,
        "DBI": agg_levels,
        "UBI": agg_levels,
    })

    # Iterate through buffer distances
    aggregations = []
    for distance in buffer_distances:
        # Create buffers for all points
        progbar.set_description(f"Calculating Sentinel2 Satellite Features for buffer {distance}m")
        data_geometry["buffer"] = data_geometry.geometry.buffer(distance)

        # Calculate bounding box of buffered points
        minx, miny, maxx, maxy = data_geometry["buffer"].total_bounds

        # Subset raster by bounding box
        subset_raster = sentinel_raster.rio.clip_box(minx, miny, maxx, maxy)

        # Convert subset raster to DataFrame
        band_coords = subset_raster.to_dataframe(name="pixel_value").reset_index()
        band_coords = band_coords.pivot_table(index=["y", "x"], columns="band", values="pixel_value").reset_index()
        band_coords.columns.name = None
        band_coords.columns = ["y", "x"] + band_names
        band_coords = gpd.GeoDataFrame(band_coords, geometry=gpd.points_from_xy(band_coords["x"], band_coords["y"]), crs=utm_crs)

        # Reduce float precision
        band_coords[band_names] = band_coords[band_names].astype(np.float32)

        # Calculate spectral indices
        band_coords["NDVI"] = (band_coords['B08'] - band_coords['B04']) / (band_coords['B08'] + band_coords['B04'])
        band_coords["BSI"] = ((band_coords['B11'] + band_coords['B04']) - (band_coords['B08'] + band_coords['B02'])) / ((band_coords['B11'] + band_coords['B04']) + (band_coords['B08'] + band_coords['B02']))
        band_coords["BSI2"] = (band_coords["B04"] - band_coords["B11"]) / (band_coords["B04"] + band_coords["B11"])
        band_coords["NDBI"] = (band_coords['B11'] - band_coords['B08']) / (band_coords['B11'] + band_coords['B08'])
        band_coords["NBSI"] = (band_coords['B04'] - band_coords['B11']) / (band_coords['B04'] + band_coords['B11'])
        band_coords["UI"] = (band_coords['B11'] - band_coords['B08']) / (band_coords['B11'] + band_coords['B08'] + band_coords['B04'])
        band_coords["NDWI"] = (band_coords['B03'] - band_coords['B08']) / (band_coords['B03'] + band_coords['B08'])
        band_coords["AWEI"] = 4 * ((band_coords['B03'] - band_coords['B12']) - (0.25 * band_coords['B08'] + 2.75 * band_coords['B11']))
        band_coords["MNDWI"] = (band_coords['B03'] - band_coords['B08']) / (band_coords['B03'] + band_coords['B08'])
        band_coords["SAVI"] = (1.5 * (band_coords['B08'] - band_coords['B04'])) / (band_coords['B08'] + band_coords['B04'] + 0.5)
        band_coords["EVI"] = 2.5 * ((band_coords['B08'] - band_coords['B04']) / (band_coords['B08'] + 6 * band_coords['B04'] - 7.5 * band_coords['B02'] + 1))
        band_coords["NDBal"] = (band_coords['B11'] - band_coords['B12']) / (band_coords['B11'] + band_coords['B12'])
        band_coords["ISA"] = 1 - band_coords["NDVI"]
        band_coords["NDISI"] = ((band_coords["B11"] - band_coords["B02"]) - (band_coords["NDVI"] + band_coords["NDWI"])) / ((band_coords["B11"] - band_coords["B02"]) + (band_coords["NDVI"] + band_coords["NDWI"]))
        band_coords["NBI"] = band_coords["B11"] / band_coords["B08"]
        band_coords["DBI"] = band_coords["NDBI"] - band_coords["NDVI"]
        band_coords["UBI"] = (band_coords["B11"] - band_coords["B04"]) / (band_coords["B11"] + band_coords["B04"])

        # Perform spatial join: pixels within buffers
        buffer_gdf = data_geometry.set_geometry("buffer")
        joined = gpd.sjoin(band_coords, buffer_gdf, how="left", predicate="intersects")

        # Aggregate statistics for all features
        grouped = joined.groupby("index_right")
        for feature, funcs in agg_funcs.items():
            agg = grouped[feature].agg(funcs)
            agg.columns = [f"{func}_{feature}_{distance}m" for func in funcs]
            aggregations.append(agg)

        # Update progress bar
        progbar.update(1)

        # Collect garbage
        gc.collect()

    # Merge all aggregations back to the original dataframe
    for agg in aggregations:
        data = pd.merge(data, agg, left_index=True, right_index=True, how="left")

    progbar.close()
    
    return data

def calculate_emissions_features(data_df, geojson_path, buffer_distances, agg_levels):
    data = data_df.copy()

    # Progress bar
    progbar = tqdm(total=len(buffer_distances), desc="Reading Emissions data")

    # Read building footprints
    emissions = gpd.read_file(geojson_path, crs=4326)
    emissions["ENERGY STAR Certification - Eligibility"] = emissions["ENERGY STAR Certification - Eligibility"].map({"Yes": 1, "No": 0}).astype(int)

    # Prepare train data as GeoDataFrame
    data_geometry = gpd.GeoDataFrame(
        data[["Longitude", "Latitude"]], geometry=gpd.points_from_xy(data["Longitude"], data["Latitude"]), crs=4326
    )

    # Estimate UTM CRS from data points and reproject both GeoDataFrames to this CRS
    utm_crs = data_geometry.estimate_utm_crs()
    data_geometry = data_geometry.to_crs(utm_crs)
    emissions = emissions.to_crs(utm_crs)

    # Aggregation functions
    agg_cols = emissions.columns.drop(["geometry", "ENERGY STAR Certification - Eligibility"]).to_list()
    agg_funcs = {col: agg_levels for col in agg_cols}
    agg_funcs.update({
        "ENERGY STAR Certification - Eligibility": ["sum"],
    })
    emissions[agg_cols] = emissions[agg_cols].astype(float)

    # Iterate through buffer distances
    aggregations = []
    for distance in buffer_distances:
        # Create buffers for all points
        progbar.set_description(f"Calculating Emission Features for buffer {distance}m")
        data_geometry["buffer"] = data_geometry.geometry.buffer(distance)

        # Perform spatial join: emissions within buffers
        buffer_gdf = data_geometry.set_geometry("buffer")
        joined = gpd.sjoin(emissions, buffer_gdf, how="left", predicate="intersects")

        # Aggregate statistics for all features
        grouped = joined.groupby("index_right", dropna=False)
        for feature, funcs in agg_funcs.items():
            agg = grouped[feature].agg(funcs)
            agg.columns = [f"{func}_{feature}_{distance}m" for func in funcs]
            aggregations.append(agg)

        # Update progress bar
        progbar.update(1)

    for agg in aggregations:
        data = data.merge(agg, left_index=True, right_index=True, how="left")

    progbar.close()

    return data

def calculate_traffic_features(data_df, geojson_path, buffer_distances, agg_levels):
    data = data_df.copy()

    # Progress bar
    progbar = tqdm(total=len(buffer_distances), desc="Reading Traffic data")

    # Read building footprints
    traffic = gpd.read_file(geojson_path, crs=4326)
    # traffic = traffic[["AvgWkdyDailyTraffic", "AADT", "HighHourValue", "geometry"]]

    # Prepare train data as GeoDataFrame
    data_geometry = gpd.GeoDataFrame(
        data[["Longitude", "Latitude"]], geometry=gpd.points_from_xy(data["Longitude"], data["Latitude"]), crs=4326
    )

    # Estimate UTM CRS from data points and reproject both GeoDataFrames to this CRS
    utm_crs = data_geometry.estimate_utm_crs()
    data_geometry = data_geometry.to_crs(utm_crs)
    traffic = traffic.to_crs(utm_crs)

    # Aggregation functions
    agg_cols = traffic.columns.drop(["geometry"]).to_list()
    agg_funcs = {col: agg_levels for col in agg_cols}

    # Iterate through buffer distances
    aggregations = []
    for distance in buffer_distances:
        # Create buffers for all points
        progbar.set_description(f"Calculating Traffic Features for buffer {distance}m")
        data_geometry["buffer"] = data_geometry.geometry.buffer(distance)

        # Perform spatial join: traffic within buffers
        buffer_gdf = data_geometry.set_geometry("buffer")
        joined = gpd.sjoin(traffic, buffer_gdf, how="left", predicate="intersects")

        # Aggregate statistics for all features
        grouped = joined.groupby("index_right", dropna=False)
        for feature, funcs in agg_funcs.items():
            agg = grouped[feature].agg(funcs)
            agg.columns = [f"{func}_{feature}_{distance}m" for func in funcs]
            aggregations.append(agg)

        # Update progress bar
        progbar.update(1)

    for agg in aggregations:
        data = data.merge(agg, left_index=True, right_index=True, how="left")

    progbar.close()

    return data

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    Used to map train data point to closest weather station
    """
    R = 6371 # Earth radius actually varies from 6356.752 km to 6378.137 km but 6371 km is the most common value
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

def calculate_weather_features(data_df, weather_path, agg_levels):

    MANHATTAN_WS = (40.76754, -73.96449)
    BRONX_WS = (40.87248, -73.89352)

    data = data_df.copy()
    weather_df = data_df.copy()

    weather_df = weather_df[["Longitude", "Latitude"]].copy()
    weather_df["haversine_to_bronx_ws"] = haversine_distance(weather_df["Latitude"], weather_df["Longitude"], BRONX_WS[0], BRONX_WS[1])
    weather_df["haversine_to_manhattan_ws"] = haversine_distance(weather_df["Latitude"], weather_df["Longitude"], MANHATTAN_WS[0], MANHATTAN_WS[1])

    weather_bronx = pd.read_excel(weather_path, sheet_name="Bronx").rename(columns={"Date / Time": "datetime"})
    weather_bronx["datetime"] = pd.to_datetime(weather_bronx["datetime"], format="%Y-%m-%d %H:%M:%S EDT").dt.floor("min")
    weather_bronx = weather_bronx.set_index("datetime")

    weather_manhattan = pd.read_excel(weather_path, sheet_name="Manhattan").rename(columns={"Date / Time": "datetime"})
    weather_manhattan["datetime"] = pd.to_datetime(weather_manhattan["datetime"], format="%Y-%m-%d %H:%M:%S EDT").dt.floor("min")
    weather_manhattan = weather_manhattan.set_index("datetime")

    static_stats = {
        "Air Temp at Surface [degC]": agg_levels,
        "Relative Humidity [percent]": agg_levels,
        "Avg Wind Speed [m/s]": agg_levels,
        "Wind Direction [degrees]": agg_levels,
        "Solar Flux [W/m^2]": agg_levels,
    }

    # Time of no use since prediction data doesnt have it, just get some stats here
    for col, agg_funcs in static_stats.items():
        for func in agg_funcs:
            weather_df.loc[weather_df["haversine_to_bronx_ws"] < weather_df["haversine_to_manhattan_ws"], f"{func}_{col}"] = weather_bronx[col].agg(func)
            weather_df.loc[weather_df["haversine_to_bronx_ws"] >= weather_df["haversine_to_manhattan_ws"], f"{func}_{col}"] = weather_manhattan[col].agg(func)

    # Rolling statistics and then select the relevant time range (nope doesnt make snese no?)

    weather_df["haversine_to_ws"] = np.where(weather_df["haversine_to_bronx_ws"] < weather_df["haversine_to_manhattan_ws"], weather_df["haversine_to_bronx_ws"], weather_df["haversine_to_manhattan_ws"])
    data = data.merge(weather_df.drop(columns=["haversine_to_bronx_ws", "haversine_to_manhattan_ws", "haversine_to_ws", "Longitude", "Latitude"]), left_index=True, right_index=True, how="left")
    return data