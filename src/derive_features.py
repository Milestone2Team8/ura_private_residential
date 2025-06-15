from tqdm import tqdm
import os
import pandas as pd
import numpy as np

import geopandas as gpd
from shapely.geometry import Point

def compute_st_metrics(gdf, index, spatial_radius, temporal_radius_days):
    target = gdf.iloc[index]
    target_point = target.geometry
    target_date = target.contract_date_dt

    gdf_time = gdf[
        (gdf.contract_date_dt >= (target_date - pd.Timedelta(days=temporal_radius_days))) &
        (gdf.contract_date_dt < target_date)
    ].copy()

    gdf_time['spatial_dist'] = gdf_time.geometry.distance(target_point)
    gdf_time = gdf_time[gdf_time['spatial_dist'] <= spatial_radius]

    if gdf_time.empty:
        return {
            'mean': np.nan,
            'std': np.nan,
            'count': 0
        }

    gdf_time['temporal_dist'] = (target_date - gdf_time.contract_date_dt).dt.days
    gdf_time['norm_dist'] = (gdf_time['spatial_dist'] / spatial_radius
                             ) + (gdf_time['temporal_dist'] / temporal_radius_days)
    gdf_time = gdf_time[gdf_time['norm_dist'] > 0]
    gdf_time['weight'] = 1 / (gdf_time['norm_dist'] ** 2)
    gdf_time['unit_price'] = gdf_time['target_price'] / gdf_time['area']

    weighted_sum = (gdf_time['weight'] * gdf_time['unit_price']).sum()
    sum_weights = gdf_time['weight'].sum()
    weighted_mean = weighted_sum / sum_weights if sum_weights > 0 else np.nan
    weighted_std = np.sqrt(np.average((gdf_time['unit_price'] - weighted_mean) ** 2, weights=gdf_time['weight']))

    return {
        'mean': weighted_mean,
        'std': weighted_std,
        'count': len(gdf_time)
    }

def derive_st_lag(df):
    output_path = "src/data/output/spatio_temporal_lags.csv"

    if os.path.exists(output_path):
        lag_df = pd.read_csv(output_path, index_col=0)
        lag_df.index = df.index
    else:
        df['geometry'] = df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
        gdf = gpd.GeoDataFrame(df.copy(), geometry='geometry')
        gdf = gdf.set_crs(epsg=4326).to_crs(epsg=3857)
        gdf['contract_date_dt'] = pd.to_datetime(gdf['contract_date_dt'])

        configs = [
            (30, 400),
            (90, 400),
            (180, 400),
            (360, 400)
        ]

        result_dict = {f'st_lag_mean_{d}d_{r}m': [] for d, r in configs}
        result_dict.update({f'st_lag_std_{d}d_{r}m': [] for d, r in configs})
        result_dict.update({f'st_neighbor_count_{d}d_{r}m': [] for d, r in configs})

        for i in tqdm(range(len(gdf)), desc="Computing multi-window ST-Lags"):
            for (days, radius) in configs:
                metrics = compute_st_metrics(gdf, i, spatial_radius=radius, temporal_radius_days=days)
                result_dict[f'st_lag_mean_{days}d_{radius}m'].append(metrics['mean'])
                result_dict[f'st_lag_std_{days}d_{radius}m'].append(metrics['std'])
                result_dict[f'st_neighbor_count_{days}d_{radius}m'].append(metrics['count'])

        lag_df = pd.DataFrame(result_dict, index=gdf.index)
        lag_df.to_csv(output_path)

    df_merged = pd.concat([df, lag_df], axis=1)
    return df_merged