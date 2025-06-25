# Background
This study aims to understand and predict private condominium prices in Singapore using five years of transactional data (2020 to 2025). This will assist potential buyers in identifying key factors influencing property prices and support them in their decision-making process. The motivation for this study arises from the relative lack of studies focused specifically on private condominiums in Singapore. To establish a clearer understanding of the data, we first performed outlier detection and clustering. Outlier detection uncovered anomalous bulk sales, which were excluded from subsequent supervised learning, while clustering revealed natural groupings of transactions based on location. Supervised learning results showed that unit area had a substantially greater influence on prices compared to other features.

Please refer to the Project Report for details and links to the data sources.

# Run the Pipeline
Execute `python run.py` to perform to perform the following steps:
1. Data cleaning and preprocessing
2. Data merging with secondary data
3. Unsupervised Learning: Outlier detection
4. Unsupervised Learning: Clustering
5. Supervised Learning: 5-fold cross validation to identify best model
6. Supervised Learning: Feature ablation analysis
7. Supervised Learning: Sensitivity analysis
8. Supervised Learning: Failure analysis

# API Setup Instructions (Optional)
To update the source URA and Google Places Points of Interest (POI) data, please follow the instructions below. These steps are optional, as the pipeline in run.py runs on the most recently downloaded data by default.

## URA API
To run src\fetch_ura_data.py, first create an URA account at https://eservice.ura.gov.sg/maps/api/#introduction to obtain an access key. Then, save this key in a .env file as URA_ACCESS_KEY="your access key".

## Google Places API
To run src\fetch_google_data.py, first create a Google Cloud account and enable the Places API at https://console.cloud.google.com/apis/library/places-backend.googleapis.com. Then, generate an API key and save it in a .env file as GOOGLE_API_KEY="your_google_places_api_key"

