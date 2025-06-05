# -- coding: utf-8 --
"""
Created on Tue May 20 11:17:01 2025

@author: nmahe
"""
import streamlit as st
import pandas as pd
import re
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import seaborn as sns

# ========== Configuration ========== 
st.set_page_config(page_title="UK Well-being & Crime Dashboard", layout="wide")

# ========== UK Well-being Dashboard ========== 
@st.cache_data
def load_wellbeing_data():
    try:
        df = pd.read_csv("Well-being_by_Quarter.csv")

        def parse_quarter_string(q_str):
            if pd.isna(q_str):
                return pd.NaT
            mapping = {
                'Jan to Mar': '01',
                'Apr to June': '04',
                'July to Sept': '07',
                'Oct to Dec': '10'
            }
            for k, v in mapping.items():
                if k in q_str:
                    match = re.search(r'\b(20\d{2})\b', q_str)
                    if match:
                        year = match.group(1)
                        return pd.to_datetime(f"{year}-{v}-01")
            return pd.NaT

        df['Quarter_Date'] = df['Quarter'].apply(parse_quarter_string)
        return df
    except Exception as e:
        st.error(f"Error loading wellbeing data: {e}")
        return pd.DataFrame()

df_wellbeing = load_wellbeing_data()

# ========== Sussex Crime Dashboard ========== 
@st.cache_data
def load_sussex_data():
    try:
        df = pd.read_csv("cleaned_sussex_crime_data.csv", parse_dates=["month"])

        required_columns = ['month', 'month_name', 'year', 'crime_type', 
                          'latitude', 'longitude', 'last_outcome_category']
        if not all(col in df.columns for col in required_columns):
            st.error("Missing essential columns in the crime dataset!")
            return pd.DataFrame()

        df["month"] = pd.to_datetime(df["month"])
        df["month_name"] = df["month"].dt.strftime('%B')
        df["year"] = df["month"].dt.year
        df["day_of_week"] = df["month"].dt.day_name()
        df["month_year"] = df["month"].dt.to_period("M").astype(str)
        return df
    except Exception as e:
        st.error(f"Error loading crime data: {e}")
        return pd.DataFrame()

df_sussex = load_sussex_data()

# ========== Main App Layout ========== 
st.title("UK Well-being & Crime Analysis Dashboard")

# ========== Well-being Section ========== 
if not df_wellbeing.empty:
    st.header("\U0001F4C8 Well-being Trends Over Time")

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_metric = st.selectbox(
            "Select a well-being metric:",
            [
                'Life Satisfaction - Mean Score',
                'Worthwhile - Mean Score',
                'Happiness - Mean Score',
                'Anxiety - Mean Score'
            ],
            key='wellbeing_metric'
        )

    fig = px.line(
        df_wellbeing.sort_values('Quarter_Date'),
        x='Quarter_Date',
        y=selected_metric,
        title=f"{selected_metric} Over Time",
        markers=True
    )
    fig.update_layout(xaxis_title="Quarter", yaxis_title="Score", height=500)
    st.plotly_chart(fig, use_container_width=True)

    with st.sidebar.expander("\U0001F4C4 Download Datasets"):
        st.download_button("Download Well-being Data as CSV", data=df_wellbeing.to_csv(index=False), file_name="wellbeing_data.csv")
        st.download_button("Download Crime Data as CSV", data=df_sussex.to_csv(index=False), file_name="crime_data.csv")

# ========== Crime Data Section ========== 
if not df_sussex.empty:
    st.header("\U0001F6A8 Sussex Crime Analysis")

    # Sidebar Filters
    st.sidebar.header("Crime Data Filters")

    available_years = sorted(df_sussex["year"].unique())
    selected_year = st.sidebar.selectbox(
        "Select Year", 
        ["All"] + available_years,
        key='year_filter'
    )

    selected_quarter = st.sidebar.selectbox(
        "Select Quarter", 
        ["All", "Q1", "Q2", "Q3", "Q4"],
        key='quarter_filter'
    )

    available_crimes = sorted(df_sussex["crime_type"].unique())
    selected_crimes = st.sidebar.multiselect(
        "Select Crime Types", 
        available_crimes,
        default=available_crimes[:3] if len(available_crimes) > 3 else available_crimes,
        key='crime_filter'
    )

    def apply_filters(df):
        filtered = df.copy()

        if selected_year != "All":
            filtered = filtered[filtered["year"] == int(selected_year)]

        if selected_quarter != "All":
            quarter_months = {
                "Q1": ["January", "February", "March"],
                "Q2": ["April", "May", "June"],
                "Q3": ["July", "August", "September"],
                "Q4": ["October", "November", "December"]
            }
            filtered = filtered[filtered["month_name"].isin(quarter_months[selected_quarter])]

        if selected_crimes:
            filtered = filtered[filtered["crime_type"].isin(selected_crimes)]

        return filtered

    filtered_df = apply_filters(df_sussex)

    if filtered_df.empty:
        st.sidebar.write("**No data available for this selection. Try adjusting the filters.**")
    else:
        available_outcomes_for_selection = sorted(filtered_df["last_outcome_category"].dropna().unique())

        if available_outcomes_for_selection:
            st.sidebar.write("**The following outcomes are available for this selection:**")
            selected_outcomes = st.sidebar.multiselect(
                "Select Outcomes",
                available_outcomes_for_selection,
                key="available_outcomes"
            )
        else:
            st.sidebar.write("**No available outcomes for this selection. Please try different filters.**")

    col1, col2 = st.columns(2)

    with col1:
        yearly_counts = filtered_df.groupby("year").size().reset_index(name="count")
        fig_yearly = px.bar(
            yearly_counts,
            x="year",
            y="count",
            title="Crimes by Year",
            labels={"year": "Year", "count": "Number of Crimes"}
        )
        st.plotly_chart(fig_yearly, use_container_width=True)

    with col2:
        crime_type_counts = filtered_df["crime_type"].value_counts().reset_index()
        crime_type_counts.columns = ["Crime Type", "Count"]
        fig_crime_types = px.pie(
            crime_type_counts,
            names="Crime Type",
            values="Count",
            title="Crime Type Distribution"
        )
        st.plotly_chart(fig_crime_types, use_container_width=True)

    st.subheader("Outcome Type Distribution")
    outcome_type_counts = filtered_df["last_outcome_category"].value_counts().reset_index()
    outcome_type_counts.columns = ["Outcome Type", "Count"]
    fig_outcome_types = px.pie(
        outcome_type_counts,
        names="Outcome Type",
        values="Count",
        title="Outcome Type Distribution"
    )
    st.plotly_chart(fig_outcome_types, use_container_width=True)

    st.subheader("Monthly Crime Trend")
    monthly_trend = filtered_df.groupby(["year", "month_name"]).size().reset_index(name="count")
    month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    monthly_trend["month_name"] = pd.Categorical(monthly_trend["month_name"], categories=month_order, ordered=True)
    monthly_trend = monthly_trend.sort_values(["year", "month_name"])

    fig_monthly = px.line(
        monthly_trend,
        x="month_name",
        y="count",
        color="year",
        title="Monthly Crime Trends by Year",
        markers=True
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

    st.subheader("Top Crime Locations")
    top_locations = filtered_df["location"].value_counts().head(10).reset_index()
    top_locations.columns = ["Location", "Crime Count"]
    fig_top_locations = px.bar(
        top_locations,
        x="Location",
        y="Crime Count",
        title="Top Crime Locations"
    )
    st.plotly_chart(fig_top_locations, use_container_width=True)

    st.subheader("Temporal Analysis: Crime by Month and Day of Week")
    temporal_counts = filtered_df.groupby(["month_name", "day_of_week"]).size().reset_index(name="crime_count")
    temporal_counts["month_name"] = pd.Categorical(temporal_counts["month_name"], categories=month_order, ordered=True)
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    temporal_counts["day_of_week"] = pd.Categorical(temporal_counts["day_of_week"], categories=days_order, ordered=True)
    fig_temporal = px.density_heatmap(
        temporal_counts, 
        x="month_name", 
        y="day_of_week", 
        z="crime_count", 
        title="Crime Heatmap by Month and Day"
    )
    st.plotly_chart(fig_temporal, use_container_width=True)

    st.subheader("Crime Locations")
    if not filtered_df.empty and "latitude" in filtered_df.columns and "longitude" in filtered_df.columns:
        map_df = filtered_df.sample(min(1000, len(filtered_df))) if len(filtered_df) > 1000 else filtered_df

        fig_map = px.scatter_mapbox(
            map_df,
            lat="latitude",
            lon="longitude",
            color="crime_type",
            hover_name="crime_type",
            zoom=10,
            height=600
        )
        fig_map.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("Crime Forecasting")
    try:
        if len(filtered_df) > 12:
            monthly_series = filtered_df.groupby("month").size()
            model = SARIMAX(monthly_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            results = model.fit(disp=False)
            forecast = results.forecast(steps=6)
            forecast_series = pd.Series(
                forecast,
                index=pd.date_range(
                    start=monthly_series.index.max() + pd.DateOffset(months=1),
                    periods=6,
                    freq="MS"
                )
            )

            fig_forecast = px.line(
                x=monthly_series.index.append(forecast_series.index),
                y=pd.concat([monthly_series, forecast_series]),
                labels={"x": "Month", "y": "Crime Count"},
                title="6-Month Crime Forecast"
            )
            fig_forecast.update_traces(mode="lines+markers")
            st.plotly_chart(fig_forecast, use_container_width=True)
        else:
            st.warning("Not enough data for forecasting (need at least 12 months)")
    except Exception as e:
        st.error(f"Forecasting error: {e}")
else:
    st.warning("No crime data available. Please check your data files.")