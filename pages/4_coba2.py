import streamlit as st
import folium
from streamlit_folium import st_folium

st.title("🗺️ GeoJSON Map of Indonesian Cities")

# ==============================
# Step 1. Create Hardcoded GeoJSON
# ==============================
geojson_data = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"city": "Jakarta"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [106.70, -6.10],
                    [106.90, -6.10],
                    [106.90, -6.30],
                    [106.70, -6.30],
                    [106.70, -6.10]
                ]]
            },
        },
        {
            "type": "Feature",
            "properties": {"city": "Bandung"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [107.55, -6.80],
                    [107.70, -6.80],
                    [107.70, -6.95],
                    [107.55, -6.95],
                    [107.55, -6.80]
                ]]
            },
        },
        {
            "type": "Feature",
            "properties": {"city": "Surabaya"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [112.60, -7.20],
                    [112.90, -7.20],
                    [112.90, -7.40],
                    [112.60, -7.40],
                    [112.60, -7.20]
                ]]
            },
        }
    ]
}

# ==============================
# Step 2. Create Map
# ==============================
m = folium.Map(location=[-2.5, 118], zoom_start=5, tiles="CartoDB positron")

# Add polygons with city name tooltips
folium.GeoJson(
    geojson_data,
    name="Indonesian Cities",
    tooltip=folium.features.GeoJsonTooltip(fields=["city"])
).add_to(m)

folium.LayerControl().add_to(m)

# ==============================
# Step 3. Show in Streamlit
# ==============================
st_folium(m, width=800, height=600)
