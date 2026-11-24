# Victor Ogunkunle: Interactive Dashboard

This dashboard shows the Rain in Australia data and predicts whether it will rain tomorrow.

## What is needed

1. Toyo's cleaned data file, weather_clean.csv. Put it in a folder called data in the main project folder.
2. Leonel's trained model file, xgboost.joblib. Put it in a folder called models in the main project folder. If the file is missing, the dashboard has a button that trains the model in about a minute.

## How to run it

Open a terminal in the main project folder and run:

    pip install streamlit plotly xgboost scikit-learn pandas joblib
    streamlit run dashboard/dashboard.py

The dashboard opens in a web browser.

## What the dashboard shows

The filters on the left are for choosing locations, years and months. They change every chart.

Chart 6 is a scatter plot that compares two weather measurements and shows which days were followed by rain.

Chart 7 is a bar chart that shows how often it rained the next day for each group, such as cloud cover or location.

Chart 8 is a histogram with a slider. Picking a range of values shows how often it rained the next day in that range.

The Rain Predictor takes today's weather and gives the chance of rain tomorrow.

The Model tab shows how well Leonel's model performed.
