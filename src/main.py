import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from io import BytesIO

sns.set_style("darkgrid")
plt.rc("figure", figsize=(16, 12))
plt.rc("font", size=13)


def display_series(series: pd.Series):
    st.text("Series-")
    st.write(series)
    st.text("Series Statistics-")
    st.write(series.describe())


def export_xlsx(df: pd.DataFrame, label: str, filename: str):
    def to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output) as xlw:
            df.to_excel(excel_writer=xlw, index=True, sheet_name="Sheet1")
        processed_data = output.getvalue()
        return processed_data

    st.download_button(
        label=label,
        data=to_excel(df),
        file_name=filename,
    )


def seasonal_decomposition(series: pd.Series):
    res = STL(series, robust=True).fit()

    df = pd.DataFrame(
        {
            "observed": res.observed,
            "seasonal": res.seasonal,
            "trend": res.trend,
            "resid": res.resid,
            "weights": res.weights,
            "ds": series.index.values,
        }
    )
    df.set_index("ds", inplace=True)

    fig = res.plot()
    st.pyplot(fig)

    return df


st.write("# Time Series Analysis")

uploaded_file = st.file_uploader(label="Choose a csv/ excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.type == "csv":
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.text("Your Data:")
    st.write(df)

    numeric_cols = list(df.select_dtypes([np.number]).columns)
    date_cols = list(df.select_dtypes(include=["datetime64"]).columns)

    if len(numeric_cols) and len(date_cols):
        col_date = st.selectbox(label="Choose date column (ds)", options=date_cols)
        col_series = st.selectbox(
            label="Choose series column (y)", options=numeric_cols
        )

        series = pd.Series(df[col_series].values, index=df[col_date], name=col_series)
        display_series(series=series)
        df = seasonal_decomposition(series=series)
        export_xlsx(
            df=df,
            label="📥 Download Seasonal Decomposition",
            filename="seasonal_decomposition.xlsx",
        )
    else:
        st.warning(
            f"""Your data should contain atleast one numeric and one datetime column
            
            Numeric Cols: {numeric_cols}
            Datetime Cols: {date_cols}
            """
        )
