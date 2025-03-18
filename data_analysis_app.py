import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Custom CSS for professional styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
    }
    h1 {
        color: #4a90e2;
        text-align: center;
    }
    .stButton button {
        background-color: #4a90e2;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stFileUploader {
        margin-bottom: 20px;
    }
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App title and description
st.title("📊 Data Analysis Tool")
st.markdown("""
    Welcome to the **Data Analysis Tool**! Upload your dataset (CSV or Excel) to perform analysis, 
    visualize data, and generate insights.
""")

# File upload
uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx"], key="file_uploader")

# Load data
def load_data(file):
    """Load data from uploaded file."""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            st.error("Unsupported file format! Please upload a CSV or Excel file.")
            return None
        st.success("✅ File uploaded successfully!")
        return df
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        return None

# Clean data
def clean_data(df):
    """Clean data: handle missing values, duplicates, and format conversion."""
    try:
        df.fillna(method='ffill', inplace=True)  # Fill missing values
        df.drop_duplicates(inplace=True)  # Remove duplicates
        
        # Convert date format if 'date' column exists
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        st.success("✅ Data cleaning completed!")
        return df
    except Exception as e:
        st.error(f"❌ Data cleaning failed: {e}")
        return df

# Analyze data
def analyze_data(df):
    """Perform statistical analysis."""
    try:
        st.subheader("📊 Basic Statistics")
        st.write(df.describe())

        if 'category' in df.columns and 'value' in df.columns:
            st.subheader("📌 Mean Value by Category")
            grouped = df.groupby("category")["value"].mean()
            st.write(grouped)
    except Exception as e:
        st.error(f"❌ Data analysis failed: {e}")

# Visualize data
def visualize_data(df):
    """Visualize data."""
    try:
        st.subheader("📈 Value Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df['value'], bins=20, kde=True, ax=ax)
        st.pyplot(fig)

        if 'category' in df.columns:
            st.subheader("📊 Mean Value by Category")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=df["category"], y=df["value"], ax=ax)
            st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Data visualization failed: {e}")

# Train a regression model
def train_model(df):
    """Train a simple regression model."""
    try:
        if 'date' in df.columns and 'value' in df.columns:
            st.subheader("🤖 Training Regression Model")
            df['timestamp'] = df['date'].astype(int) // 10**9  # Convert to timestamp

            X = df[['timestamp']]
            y = df['value']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)

            score = model.score(X_test, y_test)
            st.success(f"✅ Model trained successfully, R² score: {score:.2f}")
        else:
            st.warning("⚠️ Data is missing required 'date' or 'value' columns")
    except Exception as e:
        st.error(f"❌ Model training failed: {e}")

# Main app logic
if uploaded_file:
    # Load data
    df = load_data(uploaded_file)
    if df is not None:
        # Show raw data
        st.subheader("Raw Data")
        st.write(df.head())

        # Clean data
        df = clean_data(df)

        # Analyze data
        analyze_data(df)

        # Visualize data
        visualize_data(df)

        # Train model (if applicable)
        if st.checkbox("Train Regression Model"):
            train_model(df)

        # Download cleaned data
        st.subheader("Download Cleaned Data")
        st.write("Click below to download the cleaned dataset as a CSV file.")
        st.download_button(
            label="Download CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="cleaned_data.csv",
            mime="text/csv",
        )
else:
    st.info("👈 Please upload a file to get started.")