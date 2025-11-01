import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -------------------------------
# App Title
# -------------------------------
st.title("🧩 Customer Segmentation Analysis")
st.markdown("""
This interactive app performs **Customer Segmentation** using K-Means clustering.  
Upload your customer dataset to explore insights and visualize clusters.  
Developed by **Namratha Kadam** 💡
""")

# -------------------------------
# Upload Dataset
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # -------------------------------
    # Data Preprocessing
    # -------------------------------
    st.subheader("🧹 Data Preprocessing")
    st.write("Handling missing values and scaling numerical features...")

    df = df.dropna()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numeric_cols])

    # -------------------------------
    # Elbow Method
    # -------------------------------
    st.subheader("📈 Elbow Method for Optimal Clusters")
    inertia = []
    K = range(1, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)

    fig, ax = plt.subplots()
    plt.plot(K, inertia, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    st.pyplot(fig)

    # -------------------------------
    # Select Clusters & Apply K-Means
    # -------------------------------
    st.subheader("⚙️ K-Means Clustering")
    n_clusters = st.slider("Select number of clusters", 2, 8, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_data)

    st.write("✅ Clustering Completed!")
    st.dataframe(df.head())

    # -------------------------------
    # Cluster Visualization
    # -------------------------------
    st.subheader("🎨 Cluster Visualization")

    if len(numeric_cols) >= 2:
        x_axis = st.selectbox("Select X-axis", numeric_cols)
        y_axis = st.selectbox("Select Y-axis", numeric_cols[1:])
        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=df, x=x_axis, y=y_axis, hue='Cluster', palette='viridis')
        plt.title('Customer Segmentation Clusters')
        st.pyplot(fig2)
    else:
        st.warning("Not enough numerical columns to visualize clusters.")

    # -------------------------------
    # Cluster Summary
    # -------------------------------
    st.subheader("📋 Cluster Summary")
    summary = df.groupby('Cluster')[numeric_cols].mean()
    st.dataframe(summary.style.highlight_max(axis=0, color='lightgreen'))
else:
    st.info("👆 Please upload a CSV file to begin.")
