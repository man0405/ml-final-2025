import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
from utils.toc import Toc

import os
# Add the parent directory to the system path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


st.set_page_config(page_title="Housing Data Analysis",
                   page_icon=":chart_with_upwards_trend:",
                   layout="wide",
                   initial_sidebar_state="expanded")

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


def _max_width_(prcnt_width: int = 70):
    max_width_str = f"max-width: {prcnt_width}%;"
    st.markdown(f"""
                <style>
                .appview-container .main .block-container{{{max_width_str}}}
                </style>
                """,
                unsafe_allow_html=True,
                )


_max_width_(70)
toc = Toc()

# Placeholder for the table of contents
toc.placeholder(sidebar=True)

# Title and navigation button
col_title, col_nav = st.columns([3, 1])
with col_title:
    toc.header("🧪 Phân tích dữ liệu gốc")

with col_nav:
    if st.button("Dự đoán giá nhà", use_container_width=True):
        st.switch_page("housing_app.py")

# read data csv


@st.cache_data
def read_data(file_path: str):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        return None


data = read_data("data/vietnam_housing_dataset_final.csv")
# Drop index column if exists
data = data.drop(columns=['Unnamed: 0'], errors='ignore')
# Drop latitude and longitude if exists
data = data.drop(columns=['longitude', 'latitude',
                 'price_per_m2'], errors='ignore')
if data is not None:
    st.write("📋 Dữ liệu gốc:")
    st.dataframe(data)

    # Display basic statistics and missing values side by side
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📊 Thống kê mô tả")
        st.write(data.describe())

    with col2:
        st.subheader("❓ Giá trị thiếu")
        missing_values = data.isnull().sum()
        st.write(missing_values[missing_values > 0])
    # Display data types
    with col3:
        st.subheader("🔠 Kiểu dữ liệu")
        data_types = data.dtypes
        st.write(data_types)
# display phần phối giá nhà
    toc.subheader("📊 Phân phối giá nhà (Histogram & Boxplot)")
    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        sns.histplot(data["Price"], bins=30,
                     kde=True, ax=ax1, color="skyblue")
        ax1.set_title("Histogram giá nhà")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        sns.boxplot(y=data["Price"], ax=ax2, color="orange")
        ax2.set_title("Boxplot giá nhà")
        st.pyplot(fig2)
        # Outlier Detection and Removal

# Calculate IQR for Price column
    toc.subheader("📊 Phát hiện và loại bỏ outliers")
    Q1 = data['Price'].quantile(0.25)
    Q3 = data['Price'].quantile(0.75)
    IQR = Q3 - Q1

    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = data[(data['Price'] < lower_bound) |
                    (data['Price'] > upper_bound)]
    data_clean = data[(data['Price'] >= lower_bound) &
                      (data['Price'] <= upper_bound)]

    # Display information about outliers
    col1, col2 = st.columns(2)
    with col1:
        st.success(
            f"Số lượng outliers: {len(outliers)} ({len(outliers)/len(data):.2%} của dữ liệu)")
        st.write(f"Giới hạn dưới: {lower_bound:,.0f}")
        st.write(f"Giới hạn trên: {upper_bound:,.0f}")

    with col2:
        # Compare before and after distributions
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=data, x="Price", color="skyblue",
                     label="Trước khi xóa", alpha=0.7, kde=True, ax=ax)
        sns.histplot(data=data_clean, x="Price", color="red",
                     label="Sau khi xóa", alpha=0.7, kde=True, ax=ax)
        plt.legend()
        plt.title("Phân phối giá nhà trước và sau khi xóa outliers")
        st.pyplot(fig)
# Xử lý address qua các bước sau
    toc.header("📍 Xử lý địa chỉ")
    st.success(
        "Địa chỉ sẽ được chuyển đổi thành tọa độ (latitude, longitude) để phân tích tiếp theo.")
    data_geo = read_data("data/vietnam_housing_dataset_final.csv")
    data_train = read_data("data/data_train.csv")

    data_geo = data_geo.drop(
        columns=['Unnamed: 0', 'Address'], errors='ignore')
    data_geo_copy = data_geo.copy().drop(
        columns=['price_per_m2'], errors='ignore')
    # display data_geo_copy
    if data_geo_copy is not None:
        st.write("📋 Dữ liệu địa lý:")
        st.dataframe(data_geo_copy)
    st.success(
        "Từ tọa độ (latitude, longitude) sẽ tìm được giá nhà trung bình quanh khu vực đó.")
    if data_geo is not None:
        st.write("📋 Dữ liệu địa lý:")
        data_geo = data_geo.drop(
            columns=['latitude', 'longitude', 'price_per_m2'], errors='ignore')
        # add columns avg_price_area
        # Check if avg_price_area exists in data_train before adding it
        if data_train is not None and 'avg_price_area' in data_train.columns:
            data_geo['avg_price_area'] = data_train['avg_price_area']
        else:
            st.warning(
                "Column 'avg_price_area' not found in data_train.csv. Skipping this step.")
            st.info("Available columns in data_train: " + str(list(data_train.columns)
                    if data_train is not None else "No data loaded"))

        st.dataframe(data_geo)
# xử lý missing values của House direction vaf balcony direction,
    toc.header("🏠 Xử lý hướng nhà và ban công")
    # số liệu các giá trị duy nhất của House direction và Balcony direction
    house_directions = data_geo['House direction'].value_counts()
    balcony_directions = data_geo['Balcony direction'].value_counts()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Hướng nhà")
        st.write(house_directions)
    with col2:
        st.subheader("Hướng ban công")
        st.write(balcony_directions)
    # số lượng missing values của House direction và Balcony direction
    missing_house_direction = data_geo['House direction'].isnull().sum()
    missing_balcony_direction = data_geo['Balcony direction'].isnull().sum()
    st.write(
        f"Số lượng missing values của House direction: {missing_house_direction}")
    st.write(
        f"Số lượng missing values của Balcony direction: {missing_balcony_direction}")
    # quyết định xóa 2 column này vì số lượng missing values quá lớn
    st.warning(
        "Số lượng missing values của House direction và Balcony direction quá lớn, sẽ xóa 2 cột này.")
    data_geo = data_geo.drop(columns=['House direction', 'Balcony direction'])
    # hiển thị lại dữ liệu sau khi xóa
    st.write("📋 Dữ liệu sau khi xóa:")
    st.dataframe(data_geo)

# Xử lý Frontage và Access Road
    toc.header("🛣️ Xử lý Frontage và Access Road")
    # Hiển thị số liệu duy nhất của Frontage và Access Road
    col1, col2 = st.columns(2)
    with col1:
        toc.subheader("Frontage")
        st.write(data_geo['Frontage'].value_counts())
    with col2:
        toc.subheader("Access Road")
        st.write(data_geo['Access Road'].value_counts())
    # Số lượng missing values của Frontage và Access Road
    missing_frontage = data_geo['Frontage'].isnull().sum()
    missing_access_road = data_geo['Access Road'].isnull().sum()
    st.write(f"Số lượng missing values của Frontage: {missing_frontage}")
    st.write(f"Số lượng missing values của Access Road: {missing_access_road}")
    # Biểu đồ phân phối của Frontage và Access Road và line giá trị của nhà trung bình nếu cùng trong một Frontage và Access Road

    # Tạo data_geoFrame chứa giá trung bình theo Frontage
    frontage_avg_price = data_geo.groupby(
        'Frontage')['Price'].mean().reset_index()

    # Biểu đồ phân phối Frontage và giá trung bình
    toc.subheader("Phân phối của Frontage và giá trung bình")
    fig_frontage, ax_frontage = plt.subplots(figsize=(12, 6))

    # Vẽ countplot (phân phối)
    sns.countplot(x='Frontage', data=data, ax=ax_frontage,
                  alpha=0.7, color='skyblue')
    ax_frontage.set_title('Phân phối của Frontage và giá nhà trung bình')
    ax_frontage.set_xlabel('Frontage')
    ax_frontage.set_ylabel('Số lượng')

    # Tạo trục thứ hai cho giá trung bình
    ax2 = ax_frontage.twinx()

    # Vẽ line plot cho giá trung bình
    sns.lineplot(data=frontage_avg_price, x='Frontage', y='Price',
                 ax=ax2, color='red', marker='o', linewidth=2)
    ax2.set_ylabel('Giá trung bình (VND)', color='red')
    ax2.tick_params(axis='y', colors='red')

    # Thêm giá trị trung bình lên đường line
    for x, y in zip(frontage_avg_price['Frontage'], frontage_avg_price['Price']):
        ax2.annotate(f'{y:,.0f}', xy=(x, y), xytext=(0, 10),
                     textcoords='offset points', ha='center', va='bottom',
                     color='red', fontsize=8, rotation=45)

    plt.tight_layout()
    st.pyplot(fig_frontage)

    # Tạo DataFrame chứa giá trung bình theo Access Road
    access_road_avg_price = data_geo.groupby(
        'Access Road')['Price'].mean().reset_index()

    # Biểu đồ phân phối Access Road và giá trung bình
    toc.subheader("Phân phối của Access Road và giá trung bình")
    fig_access, ax_access = plt.subplots(figsize=(12, 6))

    # Vẽ countplot (phân phối)
    sns.countplot(x='Access Road', data=data_geo, ax=ax_access,
                  alpha=0.7, color='lightgreen')
    ax_access.set_title('Phân phối của Access Road và giá nhà trung bình')
    ax_access.set_xlabel('Access Road')
    ax_access.set_ylabel('Số lượng')

    # Tạo trục thứ hai cho giá trung bình
    ax2_access = ax_access.twinx()

    # Vẽ line plot cho giá trung bình
    sns.lineplot(data=access_road_avg_price, x='Access Road', y='Price',
                 ax=ax2_access, color='darkgreen', marker='o', linewidth=2)
    ax2_access.set_ylabel('Giá trung bình (VND)', color='darkgreen')
    ax2_access.tick_params(axis='y', colors='darkgreen')

    # Thêm giá trị trung bình lên đường line
    for x, y in zip(access_road_avg_price['Access Road'], access_road_avg_price['Price']):
        ax2_access.annotate(f'{y:,.0f}', xy=(x, y), xytext=(0, 10),
                            textcoords='offset points', ha='center', va='bottom',
                            color='darkgreen', fontsize=8, rotation=45)

    plt.tight_layout()
    st.pyplot(fig_access)
    # Biểu đồ phân phối giá của Frontage và Access Road bị missing
    st.subheader("Phân phối giá của Frontage và Access Road bị missing")
    missing_frontage_data = data[data['Frontage'].isnull()]
    missing_access_road_data = data[data['Access Road'].isnull()]
    fig_missing, ax_missing = plt.subplots(1, 2, figsize=(14, 6))
    sns.histplot(missing_frontage_data['Price'], bins=30, kde=True,
                 ax=ax_missing[0], color='skyblue')
    ax_missing[0].set_title(
        "Phân phối giá nhà khi Frontage bị missing")
    ax_missing[0].set_xlabel("Giá nhà (VND)")
    ax_missing[0].set_ylabel("Tần suất")
    sns.histplot(missing_access_road_data['Price'], bins=30, kde=True,
                 ax=ax_missing[1], color='lightgreen')
    ax_missing[1].set_title(
        "Phân phối giá nhà khi Access Road bị missing")
    ax_missing[1].set_xlabel("Giá nhà (VND)")
    ax_missing[1].set_ylabel("Tần suất")
    plt.tight_layout()
    st.pyplot(fig_missing)
    # Xử lý missing values của Frontage và Access Road
    st.markdown("### 🛠️ Xử lý missing values của Frontage và Access Road"
                )
    st.warning(
        "Số lượng missing values của Frontage và Access Road quá lớn và không có cơ sở để fill data, sẽ xóa 2 cột này.")
    data_geo = data_geo.drop(columns=['Frontage', 'Access Road'])
    # Hiển thị lại dữ liệu sau khi xóa
    st.write("📋 Dữ liệu sau khi xóa:")
    st.dataframe(data_geo)

# Xử lý Floor
    toc.header("🏢 Xử lý số tầng (Floor)"
               )
    # Hiển thị số liệu duy nhất của Floor
    toc.subheader("Số tầng (Floor)")
    st.write(data_geo['Floors'].value_counts())
    # Số lượng missing values của Floor
    missing_floor = data_geo['Floors'].isnull().sum()
    st.write(f"Số lượng missing values của Floor: {missing_floor}")
    # Biểu đồ phân phối của Floor và line giá trị của nhà trung bình nếu cùng trong một Floor
    # Tạo DataFrame chứa giá trung bình theo Floor
    floor_avg_price = data_geo.groupby(
        'Floors')['Price'].mean().reset_index()
    # Biểu đồ phân phối Floor và giá trung bình
    toc.subheader("Phân phối của số tầng và giá trung bình")
    fig_floor, ax_floor = plt.subplots(figsize=(12, 6))
    # Vẽ countplot (phân phối)
    sns.countplot(x='Floors', data=data_geo, ax=ax_floor,
                  alpha=0.7, color='skyblue')
    ax_floor.set_title('Phân phối của số tầng và giá nhà trung bình')
    ax_floor.set_xlabel('Số tầng')
    ax_floor.set_ylabel('Số lượng')
    # Tạo trục thứ hai cho giá trung bình
    ax2_floor = ax_floor.twinx()
    # Vẽ line plot cho giá trung bình
    sns.lineplot(data=floor_avg_price, x='Floors', y='Price',
                 ax=ax2_floor, color='red', marker='o', linewidth=2)
    ax2_floor.set_ylabel('Giá trung bình (VND)', color='red')
    ax2_floor.tick_params(axis='y', colors='red')
    # Thêm giá trị trung bình lên đường line
    for x, y in zip(floor_avg_price['Floors'], floor_avg_price['Price']):
        ax2_floor.annotate(f'{y:,.0f}', xy=(x, y), xytext=(0, 10),
                           textcoords='offset points', ha='center', va='bottom',
                           color='red', fontsize=8, rotation=45)
    plt.tight_layout()
    st.pyplot(fig_floor)
    # Biểu đồ phân phối giá của Floor bị missing
    toc.subheader("Phân phối giá của số tầng bị missing")
    missing_floor_data = data_geo[data_geo['Floors'].isnull()]
    fig_missing_floor, ax_missing_floor = plt.subplots(figsize=(8, 6))
    sns.histplot(missing_floor_data['Price'], bins=30, kde=True,
                 ax=ax_missing_floor, color='skyblue')
    ax_missing_floor.set_title("Phân phối giá nhà khi số tầng bị missing")
    ax_missing_floor.set_xlabel("Giá nhà (VND)")
    ax_missing_floor.set_ylabel("Tần suất")
    plt.tight_layout()
    st.pyplot(fig_missing_floor)
    # Xử lý missing values của Floor
    toc.subheader("🛠️ Xử lý missing values của số tầng")

    # Tạo bản sao để xử lý
    data_floor_processed = data_geo.copy()

    # 1. Phân chia giá nhà thành các bins (khoảng giá: 0–3, 3–6, 6–9, v.v.)
    toc.header_h4("1. Phân chia giá nhà thành các bins")

    # Tạo bins theo tỷ VND (0-3, 3-6, 6-9, ...)
    max_price = data_floor_processed['Price'].max()
    bin_size = 3  # 3 tỷ VND mỗi bin
    bins = list(range(0, int(max_price) + bin_size, bin_size))

    # Tạo labels cho bins
    bin_labels = []
    for i in range(len(bins)-1):
        bin_labels.append(f"{bins[i]}-{bins[i+1]} tỷ")

    # Thêm cột price_bin
    data_floor_processed['price_bin'] = pd.cut(data_floor_processed['Price'],
                                               bins=bins,
                                               labels=bin_labels,
                                               include_lowest=True)

    # Hiển thị phân phối của bins
    st.write("**Phân phối giá nhà theo bins:**")
    bin_counts = data_floor_processed['price_bin'].value_counts().sort_index()
    st.write(bin_counts)

    # Biểu đồ phân phối bins
    fig_bins, ax_bins = plt.subplots(figsize=(12, 6))
    bin_counts.plot(kind='bar', ax=ax_bins, color='skyblue')
    ax_bins.set_title('Phân phối số lượng nhà theo khoảng giá')
    ax_bins.set_xlabel('Khoảng giá (tỷ VND)')
    ax_bins.set_ylabel('Số lượng')
    ax_bins.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig_bins)

    # 2. Với mỗi bin, tính mode hoặc median số tầng của các bản ghi đã biết
    toc.header_h4("2. Tính toán thống kê số tầng cho mỗi bin")

    # Tính mode và median cho mỗi bin (chỉ với các giá trị không null)
    bin_stats = []

    for bin_name in bin_labels:
        bin_data = data_floor_processed[data_floor_processed['price_bin'] == bin_name]
        floors_in_bin = bin_data['Floors'].dropna()

        if len(floors_in_bin) > 0:
            # Tính mode (giá trị xuất hiện nhiều nhất)
            mode_value = floors_in_bin.mode()
            mode = mode_value.iloc[0] if len(
                mode_value) > 0 else floors_in_bin.median()

            # Tính median
            median = floors_in_bin.median()

            # Đếm số records có và thiếu floor
            total_records = len(bin_data)
            known_floors = len(floors_in_bin)
            missing_floors = total_records - known_floors

            bin_stats.append({
                'Bin': bin_name,
                'Total_Records': total_records,
                'Known_Floors': known_floors,
                'Missing_Floors': missing_floors,
                'Mode': mode,
                'Median': median,
                'Fill_Value': mode  # Sử dụng mode để fill
            })
        else:
            # Nếu không có dữ liệu floor nào trong bin này, dùng median tổng thể
            overall_median = data_floor_processed['Floors'].median()
            bin_stats.append({
                'Bin': bin_name,
                'Total_Records': len(bin_data),
                'Known_Floors': 0,
                'Missing_Floors': len(bin_data),
                'Mode': overall_median,
                'Median': overall_median,
                'Fill_Value': overall_median
            })

    # Hiển thị bảng thống kê
    stats_df = pd.DataFrame(bin_stats)
    st.write("**Thống kê số tầng theo từng bin:**")
    st.dataframe(stats_df)

    # 3. Gán giá trị tương ứng cho các bản ghi thiếu số tầng trong khoảng giá đó
    toc.header_h4("3. Điền giá trị thiếu cho số tầng")

    # Tạo dictionary mapping từ bin sang fill value
    bin_fill_map = dict(zip(stats_df['Bin'], stats_df['Fill_Value']))

    # Điền giá trị thiếu
    missing_floors_mask = data_floor_processed['Floors'].isnull()
    filled_count = 0

    for idx, row in data_floor_processed[missing_floors_mask].iterrows():
        bin_name = row['price_bin']
        if bin_name in bin_fill_map:
            data_floor_processed.loc[idx, 'Floors'] = bin_fill_map[bin_name]
            filled_count += 1

    st.success(f"Đã điền {filled_count} giá trị thiếu cho cột Floors")

    # So sánh trước và sau khi điền
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Trước khi điền:**")
        st.write(f"Số giá trị thiếu: {data_geo['Floors'].isnull().sum()}")
        st.write(f"Tổng số records: {len(data_geo)}")

    with col2:
        st.write("**Sau khi điền:**")
        st.write(
            f"Số giá trị thiếu: {data_floor_processed['Floors'].isnull().sum()}")
        st.write(f"Tổng số records: {len(data_floor_processed)}")

    # Biểu đồ so sánh phân phối số tầng trước và sau khi điền
    fig_compare, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Trước khi điền
    data_geo['Floors'].value_counts().sort_index().plot(
        kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('Phân phối số tầng TRƯỚC khi điền')
    ax1.set_xlabel('Số tầng')
    ax1.set_ylabel('Số lượng')

    # Sau khi điền
    data_floor_processed['Floors'].value_counts().sort_index().plot(
        kind='bar', ax=ax2, color='lightgreen')
    ax2.set_title('Phân phối số tầng SAU khi điền')
    ax2.set_xlabel('Số tầng')
    ax2.set_ylabel('Số lượng')

    plt.tight_layout()
    st.pyplot(fig_compare)

    # Hiển thị dữ liệu cuối cùng
    st.write("📋 **Dữ liệu sau khi xử lý missing values của Floors:**")
    st.dataframe(data_floor_processed)

    # Cập nhật data_geo với dữ liệu đã xử lý
    data_geo = data_floor_processed.copy()

# Xử lý số phòng tắm

    toc.header("🚽 Xử lý số phòng tắm (Bathrooms)")
    # Hiển thị số liệu duy nhất của Bathrooms
    toc.subheader("Số phòng tắm (Bathrooms)")
    st.write(data_geo['Bathrooms'].value_counts())
    # Số lượng missing values của Bathrooms
    missing_bathrooms = data_geo['Bathrooms'].isnull().sum()
    st.write(f"Số lượng missing values của Bathrooms: {missing_bathrooms}")

    # Biểu đồ phân phối của Bathrooms và line giá trị của nhà trung bình nếu cùng trong một Bathrooms
    # Tạo DataFrame chứa giá trung bình theo Bathrooms
    bathrooms_avg_price = data_geo.groupby(
        'Bathrooms')['Price'].mean().reset_index()
    # Biểu đồ phân phối Bathrooms và giá trung bình
    toc.subheader("Phân phối của số phòng tắm và giá trung bình")
    fig_bathrooms, ax_bathrooms = plt.subplots(figsize=(12, 6))
    # Vẽ countplot (phân phối)
    sns.countplot(x='Bathrooms', data=data_geo, ax=ax_bathrooms,
                  alpha=0.7, color='skyblue')
    ax_bathrooms.set_title('Phân phối của số phòng tắm và giá nhà trung bình')
    ax_bathrooms.set_xlabel('Số phòng tắm')
    ax_bathrooms.set_ylabel('Số lượng')
    # Tạo trục thứ hai cho giá trung bình
    ax2_bathrooms = ax_bathrooms.twinx()
    # Vẽ line plot cho giá trung bình
    sns.lineplot(data=bathrooms_avg_price, x='Bathrooms', y='Price',
                 ax=ax2_bathrooms, color='red', marker='o', linewidth=2)
    ax2_bathrooms.set_ylabel('Giá trung bình (VND)', color='red')
    ax2_bathrooms.tick_params(axis='y', colors='red')
    # Thêm giá trị trung bình lên đường line
    for x, y in zip(bathrooms_avg_price['Bathrooms'], bathrooms_avg_price['Price']):
        ax2_bathrooms.annotate(f'{y:,.0f}', xy=(x, y), xytext=(0, 10),
                               textcoords='offset points', ha='center', va='bottom',
                               color='red', fontsize=8, rotation=45)
    plt.tight_layout()
    st.pyplot(fig_bathrooms)
    # Biểu đồ phân phối giá của Bathrooms bị missing
    toc.subheader("Phân phối giá của số phòng tắm bị missing")
    missing_bathrooms_data = data_geo[data_geo['Bathrooms'].isnull()]
    fig_missing_bathrooms, ax_missing_bathrooms = plt.subplots(figsize=(8, 6))
    sns.histplot(missing_bathrooms_data['Price'], bins=30, kde=True,
                 ax=ax_missing_bathrooms, color='skyblue')
    ax_missing_bathrooms.set_title(
        "Phân phối giá nhà khi số phòng tắm bị missing")
    ax_missing_bathrooms.set_xlabel("Giá nhà (VND)")
    ax_missing_bathrooms.set_ylabel("Tần suất")
    plt.tight_layout()
    st.pyplot(fig_missing_bathrooms)
    # Xử lý missing values của Bathrooms
    st.markdown("### 🛠️ Xử lý missing values của số phòng tắm")
    # Tạo bản sao để xử lý
    data_bathrooms_processed = data_geo.copy()
    # 1. Phân chia giá nhà thành các bins (khoảng giá: 0–3, 3–6, 6–9, v.v.)
    toc.header_h4("1. Phân chia giá nhà thành các bins")
    # Tạo bins theo tỷ VND (0-3, 3-6, 6-9, ...)
    max_price = data_bathrooms_processed['Price'].max()
    bin_size = 1
    bins = list(range(0, int(max_price) + bin_size, bin_size))
    # Tạo labels cho bins
    bin_labels = []
    for i in range(len(bins)-1):
        bin_labels.append(f"{bins[i]}-{bins[i+1]} tỷ")
    # Thêm cột price_bin

    data_bathrooms_processed['price_bin'] = pd.cut(data_bathrooms_processed['Price'],
                                                   bins=bins,
                                                   labels=bin_labels,
                                                   include_lowest=True)
    # Hiển thị phân phối của bins
    st.write("**Phân phối giá nhà theo bins:**")
    bin_counts = data_bathrooms_processed['price_bin'].value_counts(
    ).sort_index()
    st.write(bin_counts)
    # Biểu đồ phân phối bins

    fig_bins_bathrooms, ax_bins_bathrooms = plt.subplots(figsize=(12, 6))
    bin_counts.plot(kind='bar', ax=ax_bins_bathrooms, color='skyblue')
    ax_bins_bathrooms.set_title('Phân phối số lượng nhà theo khoảng giá')
    ax_bins_bathrooms.set_xlabel('Khoảng giá (tỷ VND)')
    ax_bins_bathrooms.set_ylabel('Số lượng')
    ax_bins_bathrooms.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig_bins_bathrooms)
    # 2. Với mỗi bin, tính mode hoặc median số phòng tắm của các bản ghi đã biết
    toc.header_h4("2. Tính toán thống kê số phòng tắm cho mỗi bin")
    # Tính mode và median cho mỗi bin (chỉ với các giá trị không null)
    bin_stats_bathrooms = []

    for bin_name in bin_labels:
        bin_data = data_bathrooms_processed[data_bathrooms_processed['price_bin'] == bin_name]
        bathrooms_in_bin = bin_data['Bathrooms'].dropna()

        if len(bathrooms_in_bin) > 0:
            # Tính mode (giá trị xuất hiện nhiều nhất)
            mode_value = bathrooms_in_bin.mode()
            mode = mode_value.iloc[0] if len(
                mode_value) > 0 else bathrooms_in_bin.median()

            # Tính median
            median = bathrooms_in_bin.median()

            # Đếm số records có và thiếu bathroom
            total_records = len(bin_data)
            known_bathrooms = len(bathrooms_in_bin)
            missing_bathrooms = total_records - known_bathrooms

            bin_stats_bathrooms.append({
                'Bin': bin_name,
                'Total_Records': total_records,
                'Known_Bathrooms': known_bathrooms,
                'Missing_Bathrooms': missing_bathrooms,
                'Mode': mode,
                'Median': median,
                'Fill_Value': mode  # Sử dụng mode để fill
            })
        else:
            # Nếu không có dữ liệu bathroom nào trong bin này, dùng median tổng thể
            overall_median = data_bathrooms_processed['Bathrooms'].median()
            bin_stats_bathrooms.append({
                'Bin': bin_name,
                'Total_Records': len(bin_data),
                'Known_Bathrooms': 0,
                'Missing_Bathrooms': len(bin_data),
                'Mode': overall_median,
                'Median': overall_median,
                'Fill_Value': overall_median
            })
    # Hiển thị bảng thống kê
    stats_df_bathrooms = pd.DataFrame(bin_stats_bathrooms)
    st.write("**Thống kê số phòng tắm theo từng bin:**")
    st.dataframe(stats_df_bathrooms)
    # 3. Gán giá trị tương ứng cho các bản ghi thiếu số phòng tắm trong khoảng giá đó
    toc.header_h4("3. Điền giá trị thiếu cho số phòng tắm")
    # Tạo dictionary mapping từ bin sang fill value
    bin_fill_map_bathrooms = dict(
        zip(stats_df_bathrooms['Bin'], stats_df_bathrooms['Fill_Value']))
    # Điền giá trị thiếu
    missing_bathrooms_mask = data_bathrooms_processed['Bathrooms'].isnull()
    filled_count_bathrooms = 0
    for idx, row in data_bathrooms_processed[missing_bathrooms_mask].iterrows():
        bin_name = row['price_bin']
        if bin_name in bin_fill_map_bathrooms:
            data_bathrooms_processed.loc[idx,
                                         'Bathrooms'] = bin_fill_map_bathrooms[bin_name]
            filled_count_bathrooms += 1
    st.success(
        f"Đã điền {filled_count_bathrooms} giá trị thiếu cho cột Bathrooms")
    # So sánh trước và sau khi điền
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Trước khi điền:**")
        st.write(f"Số giá trị thiếu: {data_geo['Bathrooms'].isnull().sum()}")
        st.write(f"Tổng số records: {len(data_geo)}")
    with col2:
        st.write("**Sau khi điền:**")
        st.write(
            f"Số giá trị thiếu: {data_bathrooms_processed['Bathrooms'].isnull().sum()}")
        st.write(f"Tổng số records: {len(data_bathrooms_processed)}")
    # Biểu đồ so sánh phân phối số phòng tắm trước và sau khi điền
    fig_compare_bathrooms, (ax1_bathrooms, ax2_bathrooms) = plt.subplots(
        1, 2, figsize=(15, 6))
    # Trước khi điền
    data_geo['Bathrooms'].value_counts().sort_index().plot(
        kind='bar', ax=ax1_bathrooms, color='skyblue')
    ax1_bathrooms.set_title('Phân phối số phòng tắm TRƯỚC khi điền')
    ax1_bathrooms.set_xlabel('Số phòng tắm')
    ax1_bathrooms.set_ylabel('Số lượng')
    # Sau khi điền
    data_bathrooms_processed['Bathrooms'].value_counts().sort_index().plot(
        kind='bar', ax=ax2_bathrooms, color='lightgreen')
    ax2_bathrooms.set_title('Phân phối số phòng tắm SAU khi điền')
    ax2_bathrooms.set_xlabel('Số phòng tắm')
    ax2_bathrooms.set_ylabel('Số lượng')
    plt.tight_layout()
    st.pyplot(fig_compare_bathrooms)
    # Hiển thị dữ liệu cuối cùng
    st.write("📋 **Dữ liệu sau khi xử lý missing values của Bathrooms:**")
    st.dataframe(data_bathrooms_processed)
    # Cập nhật data_geo với dữ liệu đã xử lý
    data_geo = data_bathrooms_processed.copy()
# Xử lý số phòng ngủ

    toc.header("🛏️ Xử lý số phòng ngủ (Bedrooms)")
    # Hiển thị số liệu duy nhất của Bedrooms
    toc.subheader("Số phòng ngủ (Bedrooms)")
    st.write(data_geo['Bedrooms'].value_counts())
    # Số lượng missing values của Bedrooms
    missing_bedrooms = data_geo['Bedrooms'].isnull().sum()
    st.write(f"Số lượng missing values của Bedrooms: {missing_bedrooms}")
    # Biểu đồ phân phối của Bedrooms và line giá trị của nhà trung bình nếu cùng trong một Bedrooms
    # Tạo DataFrame chứa giá trung bình theo Bedrooms
    bedrooms_avg_price = data_geo.groupby(
        'Bedrooms')['Price'].mean().reset_index()
    # Biểu đồ phân phối Bedrooms và giá trung bình
    toc.subheader("Phân phối của số phòng ngủ và giá trung bình")
    fig_bedrooms, ax_bedrooms = plt.subplots(figsize=(12, 6))
    # Vẽ countplot (phân phối)
    sns.countplot(x='Bedrooms', data=data_geo, ax=ax_bedrooms,
                  alpha=0.7, color='skyblue')
    ax_bedrooms.set_title('Phân phối của số phòng ngủ và giá nhà trung bình')
    ax_bedrooms.set_xlabel('Số phòng ngủ')
    ax_bedrooms.set_ylabel('Số lượng')
    # Tạo trục thứ hai cho giá trung bình
    ax2_bedrooms = ax_bedrooms.twinx()
    # Vẽ line plot cho giá trung bình
    sns.lineplot(data=bedrooms_avg_price, x='Bedrooms', y='Price',
                 ax=ax2_bedrooms, color='red', marker='o', linewidth=2)
    ax2_bedrooms.set_ylabel('Giá trung bình (VND)', color='red')
    ax2_bedrooms.tick_params(axis='y', colors='red')
    # Thêm giá trị trung bình lên đường line
    for x, y in zip(bedrooms_avg_price['Bedrooms'], bedrooms_avg_price['Price']):
        ax2_bedrooms.annotate(f'{y:,.0f}', xy=(x, y), xytext=(0, 10),
                              textcoords='offset points', ha='center', va='bottom',
                              color='red', fontsize=8, rotation=45)
    plt.tight_layout()
    st.pyplot(fig_bedrooms)
    # Biểu đồ phân phối giá của Bedrooms bị missing
    toc.subheader("Phân phối giá của số phòng ngủ bị missing")
    missing_bedrooms_data = data_geo[data_geo['Bedrooms'].isnull()]
    fig_missing_bedrooms, ax_missing_bedrooms = plt.subplots(figsize=(8, 6))
    sns.histplot(missing_bedrooms_data['Price'], bins=30, kde=True,
                 ax=ax_missing_bedrooms, color='skyblue')

    ax_missing_bedrooms.set_title(
        "Phân phối giá nhà khi số phòng ngủ bị missing")
    ax_missing_bedrooms.set_xlabel("Giá nhà (VND)")
    ax_missing_bedrooms.set_ylabel("Tần suất")

    plt.tight_layout()
    st.pyplot(fig_missing_bedrooms)
    # Xử lý missing values của Bedrooms
    toc.subheader("🛠️ Xử lý missing values của số phòng ngủ")
    # Tạo bản sao để xử lý
    data_bedrooms_processed = data_geo.copy()
    # 1. Phân chia giá nhà thành các bins (khoảng giá: 0–3, 3–6, 6–9, v.v.)
    toc.header_h4("1. Phân chia giá nhà thành các bins")
    # Tạo bins theo tỷ VND (0-3, 3-6, 6-9, ...)
    max_price = data_bedrooms_processed['Price'].max()
    bin_size = 2
    bins = list(range(0, int(max_price) + bin_size, bin_size))
    # Tạo labels cho bins
    bin_labels = []
    for i in range(len(bins)-1):
        bin_labels.append(f"{bins[i]}-{bins[i+1]} tỷ")

    # Thêm cột price_bin
    data_bedrooms_processed['price_bin'] = pd.cut(data_bedrooms_processed['Price'],
                                                  bins=bins,
                                                  labels=bin_labels,
                                                  include_lowest=True)
    # Hiển thị phân phối của bins
    st.write("**Phân phối giá nhà theo bins:**")
    bin_counts = data_bedrooms_processed['price_bin'].value_counts(
    ).sort_index()
    st.write(bin_counts)
    # Biểu đồ phân phối bins
    fig_bins_bedrooms, ax_bins_bedrooms = plt.subplots(figsize=(12, 6))
    bin_counts.plot(kind='bar', ax=ax_bins_bedrooms, color='skyblue')
    ax_bins_bedrooms.set_title('Phân phối số lượng nhà theo khoảng giá')
    ax_bins_bedrooms.set_xlabel('Khoảng giá (tỷ VND)')
    ax_bins_bedrooms.set_ylabel('Số lượng')
    ax_bins_bedrooms.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig_bins_bedrooms)
    # 2. Với mỗi bin, tính mode hoặc median số phòng ngủ của các bản ghi đã biết
    toc.header_h4("2. Tính toán thống kê số phòng ngủ cho mỗi bin")
    # Tính mode và median cho mỗi bin (chỉ với các giá trị không null)
    bin_stats_bedrooms = []
    for bin_name in bin_labels:
        bin_data = data_bedrooms_processed[data_bedrooms_processed['price_bin'] == bin_name]
        bedrooms_in_bin = bin_data['Bedrooms'].dropna()

        if len(bedrooms_in_bin) > 0:
            # Tính mode (giá trị xuất hiện nhiều nhất)
            mode_value = bedrooms_in_bin.mode()
            mode = mode_value.iloc[0] if len(
                mode_value) > 0 else bedrooms_in_bin.median()

            # Tính median
            median = bedrooms_in_bin.median()

            # Đếm số records có và thiếu bedroom
            total_records = len(bin_data)
            known_bedrooms = len(bedrooms_in_bin)
            missing_bedrooms = total_records - known_bedrooms

            bin_stats_bedrooms.append({
                'Bin': bin_name,
                'Total_Records': total_records,
                'Known_Bedrooms': known_bedrooms,
                'Missing_Bedrooms': missing_bedrooms,
                'Mode': mode,
                'Median': median,
                'Fill_Value': mode  # Sử dụng mode để fill
            })
        else:
            # Nếu không có dữ liệu bedroom nào trong bin này, dùng median tổng thể
            overall_median = data_bedrooms_processed['Bedrooms'].median()
            bin_stats_bedrooms.append({
                'Bin': bin_name,
                'Total_Records': len(bin_data),
                'Known_Bedrooms': 0,
                'Missing_Bedrooms': len(bin_data),
                'Mode': overall_median,
                'Median': overall_median,
                'Fill_Value': overall_median
            })
    # Hiển thị bảng thống kê
    stats_df_bedrooms = pd.DataFrame(bin_stats_bedrooms)
    st.write("**Thống kê số phòng ngủ theo từng bin:**")
    st.dataframe(stats_df_bedrooms)
    # 3. Gán giá trị tương ứng cho các bản ghi thiếu số phòng ngủ trong khoảng giá đó
    toc.header_h4("3. Điền giá trị thiếu cho số phòng ngủ")
    # Tạo dictionary mapping từ bin sang fill value
    bin_fill_map_bedrooms = dict(
        zip(stats_df_bedrooms['Bin'], stats_df_bedrooms['Fill_Value']))
    # Điền giá trị thiếu
    missing_bedrooms_mask = data_bedrooms_processed['Bedrooms'].isnull()
    filled_count_bedrooms = 0
    for idx, row in data_bedrooms_processed[missing_bedrooms_mask].iterrows():
        bin_name = row['price_bin']
        if bin_name in bin_fill_map_bedrooms:
            data_bedrooms_processed.loc[idx,
                                        'Bedrooms'] = bin_fill_map_bedrooms[bin_name]
            filled_count_bedrooms += 1
    st.success(
        f"Đã điền {filled_count_bedrooms} giá trị thiếu cho cột Bedrooms")
    # So sánh trước và sau khi điền
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Trước khi điền:**")
        st.write(f"Số giá trị thiếu: {data_geo['Bedrooms'].isnull().sum()}")
        st.write(f"Tổng số records: {len(data_geo)}")
    with col2:
        st.write("**Sau khi điền:**")
        st.write(
            f"Số giá trị thiếu: {data_bedrooms_processed['Bedrooms'].isnull().sum()}")
        st.write(f"Tổng số records: {len(data_bedrooms_processed)}")
    # Biểu đồ so sánh phân phối số phòng ngủ trước và sau khi điền
    fig_compare_bedrooms, (ax1_bedrooms, ax2_bedrooms) = plt.subplots(
        1, 2, figsize=(15, 6))
    # Trước khi điền
    data_geo['Bedrooms'].value_counts().sort_index().plot(
        kind='bar', ax=ax1_bedrooms, color='skyblue')
    ax1_bedrooms.set_title('Phân phối số phòng ngủ TRƯỚC khi điền')
    ax1_bedrooms.set_xlabel('Số phòng ngủ')
    ax1_bedrooms.set_ylabel('Số lượng')
    # Sau khi điền
    data_bedrooms_processed['Bedrooms'].value_counts().sort_index().plot(
        kind='bar', ax=ax2_bedrooms, color='lightgreen')
    ax2_bedrooms.set_title('Phân phối số phòng ngủ SAU khi điền')
    ax2_bedrooms.set_xlabel('Số phòng ngủ')
    ax2_bedrooms.set_ylabel('Số lượng')
    plt.tight_layout()
    st.pyplot(fig_compare_bedrooms)
    # Hiển thị dữ liệu cuối cùng
    st.write("📋 **Dữ liệu sau khi xử lý missing values của Bedrooms:**")
    st.dataframe(data_bedrooms_processed)
    # Cập nhật data_geo với dữ liệu đã xử lý
    data_geo = data_bedrooms_processed.copy().drop(columns=['price_bin'])

# Xử lý Furntiure state
    toc.header("🪑 Xử lý tình trạng nội thất (Furniture state)")

    # Hiển thị số liệu duy nhất của Furniture state
    toc.subheader("Tình trạng nội thất (Furniture state)")
    furniture_state_counts = data_geo['Furniture state'].value_counts()
    st.write(furniture_state_counts)

    # Số lượng missing values của Furniture state
    missing_furniture = data_geo['Furniture state'].isnull().sum()
    st.write(
        f"Số lượng missing values của Furniture state: {missing_furniture}")

    # Xử lý missing values và encoding của Furniture state
    toc.subheader("🛠️ Xử lý và mã hóa tình trạng nội thất")

    # Tạo bản sao để xử lý
    data_furniture_processed = data_geo.copy()

    st.write("**Trước khi xử lý:**")
    before_furniture_counts = data_furniture_processed['Furniture state'].value_counts(
    )
    st.write(before_furniture_counts)

    # Chuyển đổi về chữ thường
    data_furniture_processed['Furniture state'] = data_furniture_processed['Furniture state'].str.lower(
    )

    # Mã hóa các giá trị
    st.info("""
    **Chiến lược mã hóa:**
    - Full (đầy đủ nội thất): 2
    - Basic (nội thất cơ bản): 1  
    - Missing/None (không nội thất): 0
    """)

    data_furniture_processed['Furniture state'] = data_furniture_processed['Furniture state'].replace({
        'full': 2,
        'basic': 1,
    })

    # Điền giá trị missing với 0 (không nội thất)
    data_furniture_processed['Furniture state'] = data_furniture_processed['Furniture state'].fillna(
        0)

    st.write("**Sau khi xử lý:**")
    after_furniture_counts = data_furniture_processed['Furniture state'].value_counts(
    )
    st.write(after_furniture_counts)

    # So sánh trước và sau khi xử lý
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Trước khi xử lý:**")
        st.write(
            f"Số giá trị thiếu: {data_geo['Furniture state'].isnull().sum()}")
        st.write(f"Tổng số records: {len(data_geo)}")

    with col2:
        st.write("**Sau khi xử lý:**")
        st.write(
            f"Số giá trị thiếu: {data_furniture_processed['Furniture state'].isnull().sum()}")
        st.write(f"Tổng số records: {len(data_furniture_processed)}")
    # Hiển thị dữ liệu cuối cùng
    st.write("📋 **Dữ liệu sau khi xử lý tình trạng nội thất:**")
    st.dataframe(data_furniture_processed)

    # Cập nhật data_geo với dữ liệu đã xử lý
    data_geo = data_furniture_processed.copy()

# Xử lý tình trạng pháp lý (Legal status)
    toc.header("⚖️ Xử lý tình trạng pháp lý (Legal status)")

    # Hiển thị số liệu duy nhất của Legal status
    toc.subheader("Tình trạng pháp lý (Legal status)")
    legal_status_counts = data_geo['Legal status'].value_counts()
    st.write(legal_status_counts)

    # Số lượng missing values của Legal status
    missing_legal = data_geo['Legal status'].isnull().sum()
    st.write(f"Số lượng missing values của Legal status: {missing_legal}")

    # Biểu đồ phân phối tình trạng pháp lý và giá trung bình
    legal_avg_price = data_geo.groupby('Legal status')[
        'Price'].mean().reset_index()

    # Xử lý missing values và one-hot encoding của Legal status
    toc.subheader("🛠️ Xử lý và mã hóa tình trạng pháp lý (One-Hot Encoding)")

    # Tạo bản sao để xử lý
    data_legal_processed = data_geo.copy()

    st.write("**Trước khi xử lý:**")
    before_legal_counts = data_legal_processed['Legal status'].value_counts()
    st.write(before_legal_counts)
    st.write(
        f"Số giá trị thiếu: {data_legal_processed['Legal status'].isnull().sum()}")

    # One-hot encoding cho Legal status
    st.info("""
    **Chiến lược mã hóa (One-Hot Encoding):**
    - Legal_status_have_certificate: 1 nếu "Have certificate", 0 nếu khác (bao gồm cả missing)
    - Legal_status_sales_contract: 1 nếu "Sale contract", 0 nếu khác (bao gồm cả missing)
    
    **Lưu ý:** Nếu Legal status là NAN/missing thì cả 2 cột đều = 0
    """)

    # Tạo các cột one-hot encoding
    data_legal_processed['Legal_status_have_certificate'] = data_legal_processed['Legal status'].apply(
        lambda x: 1 if x == 'Have certificate' else 0)
    data_legal_processed['Legal_status_sales_contract'] = data_legal_processed['Legal status'].apply(
        lambda x: 1 if x == 'Sale contract' else 0)

    # Xóa cột gốc
    data_legal_processed = data_legal_processed.drop(
        columns=['Legal status'], errors='ignore')

    st.write("**Sau khi xử lý:**")
    st.write("Các cột mới được tạo:")
    legal_columns = ['Legal_status_have_certificate',
                     'Legal_status_sales_contract']
    for col in legal_columns:
        if col in data_legal_processed.columns:
            st.write(
                f"- {col}: {data_legal_processed[col].value_counts().to_dict()}")

    # Biểu đồ so sánh các cột mới được tạo
    st.subheader("Phân phối các cột tình trạng pháp lý sau mã hóa")
    fig_encoded_legal, axes = plt.subplots(1, 2, figsize=(15, 6))

    for i, col in enumerate(legal_columns):
        if col in data_legal_processed.columns:
            data_legal_processed[col].value_counts().plot(
                kind='bar', ax=axes[i], color=['lightcoral', 'skyblue'])
            axes[i].set_title(f'Phân phối {col}')
            axes[i].set_xlabel('Giá trị')
            axes[i].set_ylabel('Số lượng')
            axes[i].tick_params(axis='x', rotation=0)

    plt.tight_layout()
    st.pyplot(fig_encoded_legal)

    # Tạo bảng kiểm tra
    original_legal = data_geo['Legal status'].copy()
    check_df = pd.DataFrame({
        'Original_Legal_Status': original_legal,
        'Have_Certificate': data_legal_processed['Legal_status_have_certificate'],
        'Sales_Contract': data_legal_processed['Legal_status_sales_contract']
    })
    # Hiển thị dữ liệu cuối cùng
    st.write("📋 **Dữ liệu sau khi xử lý tình trạng pháp lý:**")
    st.dataframe(data_legal_processed)

    # Cập nhật data_geo với dữ liệu đã xử lý
    data_geo = data_legal_processed.copy()


# Button để tải xuống dữ liệu đã xử lý
    toc.header("📥 Tải xuống dữ liệu đã xử lý")
    csv = data_geo.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Tải xuống dữ liệu đã xử lý (CSV)",
        data=csv,
        file_name='processed_data.csv',
        mime='text/csv',
        key='download-csv'
    )

    # Hiển thị dữ liệu cuối cùng
    st.write("📋 **Dữ liệu cuối cùng sau khi xử lý:**")
    st.dataframe(data_geo)
toc.generate()
