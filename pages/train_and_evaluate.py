import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

# Streamlit page configuration
st.set_page_config(
    page_title="Training Model và đánh giá kết quả",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to keep sidebar always open and hide collapse button
st.markdown("""
<style>
    /* Hide the sidebar collapse button */
    .css-1d391kg {
        display: none;
    }
    
    /* Alternative selector for newer Streamlit versions */
    button[kind="header"][data-testid="baseButton-header"] {
        display: none;
    }
    
    /* Ensure sidebar stays expanded */
    .css-1lcbmhc {
        min-width: 244px !important;
        max-width: 244px !important;
    }
    
    /* Alternative for newer versions */
    section[data-testid="stSidebar"] {
        min-width: 244px !important;
        max-width: 244px !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Training Model và đánh giá kết quả")
st.markdown("So sánh các mô hình máy học đã được huấn luyện trên dữ liệu gốc và dữ liệu đã chuyển đổi PCA để dự đoán giá nhà")
st.markdown("---")

# Load pre-trained results with caching


@st.cache_data
def load_training_results():
    """Load pre-trained model results"""
    try:
        results_df = pd.read_csv(
            '/Users/mandev/Workspace/UNI/ML/Final/model/training_results.csv')
        return results_df
    except FileNotFoundError:
        st.error(
            "❌ Kết quả huấn luyện mô hình không tìm thấy! Vui lòng chạy `python train_models.py` trước.")
        return None


@st.cache_data
def load_pca_analysis():
    """Load PCA analysis results"""
    try:
        with open('/Users/mandev/Workspace/UNI/ML/Final/model/pca_analysis.json', 'r') as f:
            pca_data = json.load(f)
        return pca_data
    except FileNotFoundError:
        st.error(
            "❌ Phân tích PCA không tìm thấy! Vui lòng chạy `python train_models.py` trước.")
        return None


@st.cache_data
def load_feature_names():
    """Load feature names"""
    try:
        with open('/Users/mandev/Workspace/UNI/ML/Final/model/feature_names.json', 'r') as f:
            feature_names = json.load(f)
        return feature_names
    except FileNotFoundError:
        return []


# Load all data
results_df = load_training_results()
pca_data = load_pca_analysis()
feature_names = load_feature_names()

if results_df is None or pca_data is None:
    st.stop()

# Data overview
st.header("📊 Dataset Overview")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Tổng số dòng", pca_data['data_info']['total_samples'])
with col2:
    st.metric("Số lượng đặc trưng", pca_data['data_info']['features'])
with col3:
    st.metric("Đối tượng", pca_data['data_info']['target'])

if feature_names:
    st.write("**Số lượng đặc trưng:**", ", ".join(feature_names))

# PCA Analysis Results
st.header("🧮 Phân tích kết quả")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Explained Variance by Component")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    explained_variance_ratio = np.array(pca_data['explained_variance_ratio'])
    ax1.bar(range(1, len(explained_variance_ratio) + 1),
            explained_variance_ratio)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Explained Variance by Each Principal Component')
    st.pyplot(fig1)

with col2:
    st.subheader("📈 Cumulative Explained Variance")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    cumulative_variance = np.array(pca_data['cumulative_variance'])
    ax2.plot(range(1, len(cumulative_variance) + 1),
             cumulative_variance, marker='o')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Cumulative Explained Variance')
    ax2.axhline(y=pca_data['pca_variance_threshold'], color='r', linestyle='--',
                label=f'{pca_data["pca_variance_threshold"]*100}% Variance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

# PCA Summary
st.info(
    f"** Số lượng thành phần cho {pca_data['pca_variance_threshold']*100}% phương sai:** {pca_data['n_components_selected']}")
st.success(
    f"**Giảm chiều dữ liệu:** {pca_data['original_features']} → {pca_data['n_components_selected']} features ( Giảm {((pca_data['original_features'] - pca_data['n_components_selected']) / pca_data['original_features'] * 100):.1f}% )")

# Model Training & Evaluation Results
st.header("🤖 Training Model và đánh giá kết quả")
st.success(
    "Các mô hình đã được huấn luyện và đánh giá trên dữ liệu gốc và dữ liệu đã chuyển đổi PCA. Dưới đây là kết quả chi tiết.")

# Display detailed results
st.header("📊 So sánh kết quả")

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(
    ["📋 Kết quả chi tiết", "🏆 Model tốt nhất", "📈 Visualizations"])

with tab1:
    st.subheader("Tất cả kết quả mô hình")
    st.dataframe(results_df.round(4), use_container_width=True)

    # Download results
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="model_comparison_results.csv",
        mime="text/csv"
    )

with tab2:
    st.subheader("🥇 Tổng kết model tốt nhất")

    # Best model overall
    best_overall = results_df.loc[results_df['Test R²'].idxmax()]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="🏆 Mô hình tổng thể tốt nhất",
            value=f"{best_overall['Model']} ({best_overall['Data Type']})",
            delta=f"R² = {best_overall['Test R²']:.4f}"
        )
        st.write(f"**Test MSE:** {best_overall['Test MSE']:.4f}")
        st.write(f"**Test MAE:** {best_overall['Test MAE']:.4f}")

    with col2:
        # Best model for original data
        best_original = results_df[results_df['Data Type'] == 'Original'].loc[
            results_df[results_df['Data Type'] ==
                       'Original']['Test R²'].idxmax()
        ]
        st.metric(
            label="🥇 Model tốt nhất trên data gốc",
            value=best_original['Model'],
            delta=f"R² = {best_original['Test R²']:.4f}"
        )
        st.write(f"**Test MSE:** {best_original['Test MSE']:.4f}")

    with col3:
        # Best model for PCA data
        best_pca = results_df[results_df['Data Type'] == 'PCA'].loc[
            results_df[results_df['Data Type'] == 'PCA']['Test R²'].idxmax()
        ]
        st.metric(
            label="🥇 Model tốt nhất trên PCA Data",
            value=best_pca['Model'],
            delta=f"R² = {best_pca['Test R²']:.4f}"
        )
        st.write(f"**Test MSE:** {best_pca['Test MSE']:.4f}")

with tab3:
    st.subheader("📊 So sánh hiệu suất")

    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Data for plotting
    original_r2 = results_df[results_df['Data Type'] == 'Original']['Test R²']
    pca_r2 = results_df[results_df['Data Type'] == 'PCA']['Test R²']
    model_names = results_df[results_df['Data Type'] == 'Original']['Model']

    x = np.arange(len(model_names))
    width = 0.35

    # Plot 1: R² scores comparison
    ax1.bar(x - width/2, original_r2, width,
            label='Original Data', alpha=0.8, color='skyblue')
    ax1.bar(x + width/2, pca_r2, width, label='PCA Data',
            alpha=0.8, color='lightcoral')
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Test R² Score')
    ax1.set_title('Test R² Score Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: MSE comparison
    original_mse = results_df[results_df['Data Type']
                              == 'Original']['Test MSE']
    pca_mse = results_df[results_df['Data Type'] == 'PCA']['Test MSE']

    ax2.bar(x - width/2, original_mse, width,
            label='Original Data', alpha=0.8, color='skyblue')
    ax2.bar(x + width/2, pca_mse, width, label='PCA Data',
            alpha=0.8, color='lightcoral')
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Test MSE')
    ax2.set_title('Test MSE Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: CV scores comparison
    original_cv = results_df[results_df['Data Type']
                             == 'Original']['CV R² Mean']
    pca_cv = results_df[results_df['Data Type'] == 'PCA']['CV R² Mean']

    ax3.bar(x - width/2, original_cv, width,
            label='Original Data', alpha=0.8, color='skyblue')
    ax3.bar(x + width/2, pca_cv, width, label='PCA Data',
            alpha=0.8, color='lightcoral')
    ax3.set_xlabel('Models')
    ax3.set_ylabel('Cross-Validation R² Score')
    ax3.set_title('Cross-Validation R² Score Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Performance improvement
    improvement = ((pca_r2.values - original_r2.values) /
                   original_r2.values) * 100
    colors = ['green' if x > 0 else 'red' for x in improvement]
    ax4.bar(x, improvement, color=colors, alpha=0.7)
    ax4.set_xlabel('Models')
    ax4.set_ylabel('R² Improvement (%)')
    ax4.set_title('Performance Change with PCA')
    ax4.set_xticks(x)
    ax4.set_xticklabels(model_names, rotation=45, ha='right')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

# Summary statistics
st.header("📈 Thống kê tóm tắt")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Hiệu suất trên dữ liệu gốc")
    orig_stats = results_df[results_df['Data Type'] == 'Original']['Test R²']
    st.metric("Average Test R²", f"{orig_stats.mean():.4f}")
    st.metric("Best Test R²", f"{orig_stats.max():.4f}")
    st.metric("Worst Test R²", f"{orig_stats.min():.4f}")

with col2:
    st.subheader("📊 Hiệu suất trên dữ liệu PCA")
    pca_stats = results_df[results_df['Data Type'] == 'PCA']['Test R²']
    st.metric("Average Test R²", f"{pca_stats.mean():.4f}")
    st.metric("Best Test R²", f"{pca_stats.max():.4f}")
    st.metric("Worst Test R²", f"{pca_stats.min():.4f}")
