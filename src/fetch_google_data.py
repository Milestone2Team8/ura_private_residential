"""
This module fetches POI data from the Google Places API using a grid-based approach
and visualizes the results using folium.
"""

import os
import time
import json
from pathlib import Path

import requests
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import folium
import osmnx as ox
from shapely.geometry import Point

def get_urban_polygons(place="Singapore"):
    """
    Retrieves and processes urban land use polygons from OpenStreetMap for a given place.

    Args:
        place (str): The name of the place to retrieve features for.

    Returns:
        geopandas.GeoDataFrame: A simplified GeoDataFrame representing the urban area.
    """
    tags = {
        'landuse': ['residential', 'commercial', 'industrial', 'retail']
    }

    gdf = ox.features_from_place(place, tags)
    gdf = gdf[gdf.get('natural') != 'water']
    gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]

    urban_area = gdf.dissolve()
    urban_area['geometry'] = urban_area['geometry'].simplify(tolerance=0.0005)

    return urban_area

def create_grid(urban_geom, step=0.005):
    """
    Creates a grid of lat/lng points within bounding box that fall inside urban_geom.

    Returns:
        list: List of (lat, lng) tuples
    """

    min_lat=1.22
    max_lat=1.47
    min_lng=103.6
    max_lng=104.1

    lat_range = np.arange(min_lat, max_lat, step)
    lng_range = np.arange(min_lng, max_lng, step)

    grid_points = []
    for lat in lat_range:
        for lng in lng_range:
            point = Point(lng, lat)
            if urban_geom.contains(point):
                grid_points.append((lat, lng))

    return grid_points

def g_nearby_search(g_api_key, lat, lng, radius, poi_type):
    """
    Retrieves POIs of a given type near the specified lat/lng.

    Returns:
        list: List of place result dictionaries.
    """
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius": radius,
        "type": poi_type,
        "key": g_api_key
    }

    all_results = []

    for _ in range(3):
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        results = data.get("results", [])
        all_results.extend(results)

        next_token = data.get("next_page_token")
        if not next_token:
            break

        time.sleep(2)
        params = {"pagetoken": next_token, "key": g_api_key}

    return all_results


def plot_google_poi(gdf_urban, grid_points, poi_df, output_html_path):
    m = folium.Map(location=[1.3521, 103.8198], zoom_start=11)
    folium.GeoJson(gdf_urban, name="Urban Area").add_to(m)

    for lat, lng in grid_points:
        folium.Circle(
            location=[lat, lng],
            radius=radius,
            color='red',
            fill=True,
            fill_opacity=0.3
        ).add_to(m)

    for _, row in poi_df.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lng']],
            radius=3,
            color='blue',
            fill=True,
            fill_opacity=0.7,
            popup=row['name']
        ).add_to(m)

    m.save(output_html_path)


def fetch_google_data(g_api_key, poi_type, radius=500, place="Singapore", test_run=False):
    """
    Fetches POIs of a given type from Google Places API inside urban areas of a given place.

    Returns:
        tuple:
            - pd.DataFrame: Cleaned POI results
            - list: Error coordinates
    """
    output_json_path = Path()
    output_csv_path = Path(f"./src/data/input/raw_google_data_{poi_type}.csv")
    output_html_path = Path(f"./src/data/plot/raw_google_data_{poi_type}.html")

    gdf_urban = get_urban_polygons(place=place)
    grid_points = create_grid(urban_geom= gdf_urban.geometry.values[0])

    all_results = []
    error_points = []

    for i, (lat, lng) in enumerate(grid_points):
        if i == 20 and test_run is True:
            break
        print(f"{np.round((i + 1) * 100 / len(grid_points), 2)}%")
        try:
            results = g_nearby_search(g_api_key, lat, lng, radius, poi_type)
            all_results.extend(results)
        except requests.RequestException:
            error_points.append((lat, lng))

    df = pd.DataFrame(all_results)[['place_id', 'name', 'business_status', 'geometry',
                                    'price_level', 'user_ratings_total', 'rating',
                                    'types', 'vicinity', 'permanently_closed'
                                    ]]
    df['lat'] = df['geometry'].apply(lambda x: x['location']['lat'])
    df['lng'] = df['geometry'].apply(lambda x: x['location']['lng'])

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    df.to_csv(output_csv_path, index=False)

    plot_google_poi(gdf_urban, grid_points, df, output_html_path)

    return df, error_points

if __name__ == "__main__":
    import argparse

    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(dotenv_path=env_path)

    parser = argparse.ArgumentParser(description="Fetch POI data from Google Places API")
    parser.add_argument("--poi_type", type=str, required=True,
                        help="Type of POI to fetch (e.g., restaurant, school)")
    parser.add_argument("--radius", type=int, default=500,
                        help="Search radius in meters (default: 500)")
    parser.add_argument("--place", type=str, default="Singapore",
                        help="Place to get POIs from (default: Singapore)")
    parser.add_argument("--test_run", action='store_true',
                        help="Toggle test run")

    args = parser.parse_args()

    g_api_key = os.getenv("GOOGLE_API_KEY")

    fetch_google_data(
        g_api_key=g_api_key,
        poi_type=args.poi_type,
        radius=args.radius,
        place=args.place,
        test_run=args.test_run
    )
