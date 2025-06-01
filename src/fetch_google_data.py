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
import geopandas as gpd

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

def get_ura_urban_polygons():
    """
    Fetches and processes urban planning polygon data from Singapore's URA dataset API.

    This function downloads urban area polygon data from the Singapore government's open data API, 
    parses it into a GeoDataFrame, sets the geometry and coordinate reference system (CRS), 
    dissolves individual polygons into a single unified urban area, and simplifies the geometry 
    for more efficient spatial processing.

    Returns:
        geopandas.GeoDataFrame: A simplified GeoDataFrame containing 
        the dissolved urban area geometry with EPSG:4326 as the coordinate reference system.
    """
    dataset_id = "d_4765db0e87b9c86336792efe8a1f7a66"
    url = "https://api-open.data.gov.sg/v1/public/api/datasets/" + dataset_id + "/poll-download"

    response = requests.get(url, timeout=10)
    json_data = response.json()

    url = json_data['data']['url']
    response = requests.get(url, timeout=10)
    response = response.json()

    gdf = gpd.GeoDataFrame.from_features(response['features'])
    gdf = gdf.set_geometry("geometry")
    gdf.set_crs(epsg=4326, inplace=True)

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


def plot_google_poi(gdf_urban, grid_points, poi_df, output_html_path, radius):
    """
    Plots urban area boundaries, grid search points, and POI locations on a folium map 
    and saves it as an interactive HTML file.

    Args:
        gdf_urban (geopandas.GeoDataFrame): GeoDataFrame containing the geometry of the urban area.
        grid_points (list of tuple): List of (lat, lng) tuples representing grid search points.
        poi_df (pandas.DataFrame): DataFrame of POIs with 'lat', 'lng', and 'name' columns.
        output_html_path (str or Path): Path to save the resulting HTML map file.
        radius (int): Radius in meters for drawing grid circles.

    Returns:
        None. Saves an HTML file containing the folium map.
    """

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


def fetch_google_data(g_api_key, poi_type, radius=550, 
                      test_run=False,
#                      place="Singapore"
                      ):
    """
    Fetches POIs of a given type from Google Places API inside urban areas of a given place.

    Returns:
        tuple:
            - pd.DataFrame: Cleaned POI results
            - list: Error coordinates
    """

#    gdf_urban = get_urban_polygons(place=place)
    gdf_urban = get_ura_urban_polygons()
    grid_points = create_grid(urban_geom= gdf_urban.geometry.values[0], step=0.007)

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

    with open(Path(f"./src/data/input/raw_google_data_{poi_type}.json"),
               "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    df.to_csv(Path(f"./src/data/input/raw_google_data_{poi_type}.csv"), index=False)

    plot_google_poi(gdf_urban, grid_points, df,
                    Path(f"./src/data/plot/raw_google_data_{poi_type}.html"), radius)

    return df, error_points

if __name__ == "__main__":
    import argparse

    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(dotenv_path=env_path)

    parser = argparse.ArgumentParser(description="Fetch POI data from Google Places API")
    parser.add_argument("--poi_type", type=str, required=True,
                        help="Type of POI to fetch (e.g., restaurant, school)")
    parser.add_argument("--radius", type=int, default=550,
                        help="Search radius in meters (default: 550)")
    parser.add_argument("--place", type=str, default="Singapore",
                        help="Place to get POIs from (default: Singapore)")
    parser.add_argument("--test_run", action='store_true',
                        help="Toggle test run")

    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")

    fetch_google_data(
        g_api_key=api_key,
        poi_type=args.poi_type,
        radius=args.radius,
        place=args.place,
        test_run=args.test_run
    )
