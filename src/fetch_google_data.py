import requests
import time
import json
from pathlib import Path
from dotenv import load_dotenv
import os

import osmnx as ox
import geopandas as gpd
import folium
from shapely.geometry import Point

import pandas as pd
import numpy as np

def get_urban_polygons(place="Singapore"):
    """
    Retrieves and processes urban land use polygons from OpenStreetMap for a given place.

    Args:
        place (str): The name of the place to retrieve features for (default is "Singapore").

    Returns:
        geopandas.GeoDataFrame: A dissolved and simplified GeoDataFrame representing the urban area,
                                excluding natural water bodies and including only Polygon and MultiPolygon geometries.
    """
    tags = {
        'landuse': ['residential', 'commercial', 'industrial', 'retail'],
        # 'landuse': True
    }

    gdf = ox.features_from_place(place, tags)

    gdf = gdf[gdf.get('natural') != 'water']
    gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]

    urban_area = gdf.dissolve()
    urban_area['geometry'] = urban_area['geometry'].simplify(tolerance=0.0005)

    return urban_area


def create_grid(urban_geom, min_lat = 1.22, max_lat = 1.47, min_lng = 103.6, max_lng = 104.1, step=0.005):
    """
    Creates a grid of latitude and longitude points within a bounding box and filters
    them to only include those contained within the given urban geometry.

    Args:
        min_lat (float): Minimum latitude of the grid.
        max_lat (float): Maximum latitude of the grid.
        min_lng (float): Minimum longitude of the grid.
        max_lng (float): Maximum longitude of the grid.
        urban_geom (shapely.geometry or geopandas.GeoSeries): Geometry to check point inclusion against.
        step (float): Step size for the grid in degrees (default is 0.005).

    Returns:
        list: A list of (lat, lng) tuples that fall within the specified urban geometry.
    """
    lat_range = np.arange(min_lat, max_lat, step)
    lng_range = np.arange(min_lng, max_lng, step)

    grid_points = []
    for lat in lat_range:
        for lng in lng_range:
            point = Point(lng, lat)
            if urban_geom.contains(point):
                grid_points.append((lat, lng))

    return grid_points

def g_nearby_search(api_key, lat, lng, radius, poi_type):

    """
    Retrieves nearby points of interest (POIs) of type 'restaurant' from the Google Places API.

    Args:
        api_key (str): Google Places API key.
        lat (float): Latitude of the search center.
        lng (float): Longitude of the search center.
        radius (int): Search radius in meters.
        type (str or list): Type of place to search for (ex. 'restaurant').

    Returns:
        list: A list of dictionaries, each representing a place result from the API response.
              The function automatically fetches up to 3 pages (MAX 20 Per page) of results if available.
    """
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

    params = {
        "location": f"{lat},{lng}",
        "radius": radius,
        "type": poi_type,
        "key": api_key
    }

    all_results = []

    for i in range(3):  # max 3 pages
        response = requests.get(url, params=params)
        data = response.json()

        results = data.get("results", [])
        all_results.extend(results)

        next_token = data.get("next_page_token")
        if not next_token:
            break

        time.sleep(2)
        params = {
            "pagetoken": next_token,
            "key": api_key
        }

    return all_results

def fetch_google_data(api_key, poi_type, radius = 500, place="Singapore", test_run = False):
    """
    Fetches points of interest (POIs) of a given type from the Google Places API within an urban area.

    The function first retrieves urban land use polygons for the specified place, then creates a grid 
    of points inside this area. It queries the Google Places API for each point and collects the POI data.
    Results are consolidated into a DataFrame, and any grid points that caused errors during API requests 
    are tracked.

    Args:
        api_key (str): Google Places API key.
        lat (float): Latitude used to define the initial grid area (not used directly in query).
        lng (float): Longitude used to define the initial grid area (not used directly in query).
        radius (int): Search radius in meters for each point in the grid.
        type (str or list): Type of place to search for (e.g., 'restaurant', 'school', etc.).
        place (str): Name of the place (city, country) to retrieve urban polygons for (default is "Singapore").

    Returns:
        tuple:
            - pd.DataFrame: A DataFrame containing POI results with columns such as 'place_id', 'name', 
                            'rating', 'types', etc., including extracted 'lat' and 'lng'.
            - list: A list of (lat, lng) tuples that caused errors during the API request.
    """

    output_json_path = Path(f"./src/data/input/raw_google_data_{poi_type}.json")
    output_csv_path = Path(f"./src/data/input/raw_google_data_{poi_type}.csv")
    output_html_path = Path(f"./src/data/plot/raw_google_data_{poi_type}.html")

    gdf_urban = get_urban_polygons(place=place)
    urban_geom = gdf_urban.geometry.values[0]
    grid_points = create_grid(urban_geom=urban_geom)

    all_results = []
    error_points = []

    n = 0
    for lat, lng in grid_points:
        n+=1
        if n == 20 and test_run == True:
            break
        print(f"{np.round(n * 100 / len(grid_points), 2)}%")
        try:
            results = g_nearby_search(api_key, lat, lng, radius, poi_type)
            all_results.extend(results)
        except:
            error_points.append((lat, lng))

    column_list = ['place_id', 'name', 'business_status', 'geometry',
                   'price_level', 'user_ratings_total', 'rating', 
                   'types', 'vicinity', 'permanently_closed']

    df = pd.DataFrame(all_results)[column_list]
    df.loc[:, 'lat'] = df['geometry'].apply(lambda x: x['location']['lat'])
    df.loc[:, 'lng'] = df['geometry'].apply(lambda x: x['location']['lng'])


    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    df.to_csv(output_csv_path, index=False)

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

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lng']],
            radius=3,
            color='blue',
            fill=True,
            fill_opacity=0.7,
            popup=row['name']
        ).add_to(m)

    m.save(output_html_path)

    return df, error_points


if __name__ == "__main__":
    import argparse

    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(dotenv_path=env_path)

    parser = argparse.ArgumentParser(description="Fetch POI data from Google Places API")
    parser.add_argument("--poi_type", type=str, required=True, help="Type of POI to fetch (e.g., restaurant, school)")
    parser.add_argument("--radius", type=int, default=500, help="Search radius in meters (default: 500)")
    parser.add_argument("--place", type=str, default="Singapore", help="Place to get POIs from (default: Singapore)")
    parser.add_argument("--test_run", type=bool, default=False, help="Toggle test run")

    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")

    fetch_google_data(
        api_key=api_key,
        poi_type=args.poi_type,
        radius=args.radius,
        place=args.place,
        test_run=args.test_run
    )
