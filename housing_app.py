import time
from datetime import datetime
from geopy.exc import GeocoderUnavailable
from utils.combiner import CombinedAttributesAdder
from streamlit_folium import st_folium
import geopy.distance
from geopy.geocoders import Nominatim
import folium
import pickle
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Housing Prices Prediction",
    page_icon=":house:",
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


def initialize_session_states():
    if 'markers' not in st.session_state:
        st.session_state['markers'] = []
    if 'lines' not in st.session_state:
        st.session_state['lines'] = []
    if 'fg' not in st.session_state:
        st.session_state['fg'] = None
    if 'prediction' not in st.session_state:
        st.session_state['prediction'] = None
    if 'pca_prediction' not in st.session_state:
        st.session_state['pca_prediction'] = None
    if 'address' not in st.session_state:
        st.session_state['address'] = None
    if 'address_output' not in st.session_state:
        st.session_state['address_output'] = ""
    if 'location' not in st.session_state:
        st.session_state['location'] = None
    if 'map_center' not in st.session_state:
        st.session_state['map_center'] = [14.0, 110.0]
    if 'map_zoom' not in st.session_state:
        st.session_state['map_zoom'] = 5.5
    if 'prediction_history' not in st.session_state:
        st.session_state['prediction_history'] = []


# Call initialization immediately
initialize_session_states()
# cached resources


@st.cache_resource
def initialize_nominatim(user_agent=f'housing_price_app_{np.random.randint(0, 200)}'):
    with st.spinner('Initializing geolocator...'):
        return Nominatim(user_agent=user_agent)


@st.cache_resource
def load_model(filepath: str):
    with st.spinner('Loading model...'):
        return pickle.load(open(filepath, 'rb'))


@st.cache_resource
def load_combiner():
    with st.spinner('Loading components..'):
        return CombinedAttributesAdder()

# read file csv


@st.cache_data
def read_csv_file(filepath: str):
    with st.spinner(f'Reading {filepath}...'):
        try:
            return pd.read_csv(filepath)
        except FileNotFoundError:
            st.error(f"File not found: {filepath}")
            return pd.DataFrame()


geolocator = initialize_nominatim()
loaded_model_package = load_model('model/best_original_model.pkl')
loaded_pca_model_package = load_model('model/best_pca_model.pkl')
combiner = load_combiner()
cell_filter = read_csv_file('data/vietnam_grid_5s.csv')
cell_filter['avg_price'] = cell_filter['total_price'] / cell_filter['quantity']

epsilon = 0.01389


def get_location(address: str, timeout: int = 10):
    try:
        return geolocator.geocode(address, addressdetails=True, timeout=timeout)
    except GeocoderUnavailable as e:
        st.error(
            f"Lỗi kết nối với dịch vụ định vị: {str(e)}. Vui lòng kiểm tra kết nối mạng hoặc thử lại sau vài giây.")
        return None


def transform_data(data: pd.DataFrame):
    return combiner.add_nearest_cities(data)


def get_avg_price_are(lat, lon):
    matches = cell_filter[(cell_filter['lat'] <= lat) & (lat <= cell_filter['lat'] + epsilon) &
                          (cell_filter['log'] <= lon) & (lon <= cell_filter['log'] + epsilon)]

    if len(matches) > 0:
        return matches['avg_price'].mean()
    else:
        return cell_filter['avg_price'].mean()


def process_input_for_prediction(input_data):
    """Process input data to match the model's expected format"""
    # Convert input to DataFrame
    df = pd.DataFrame([input_data])

    # Map categorical values to numerical
    # Furniture state mapping: Basic=0, Full=1, None=2
    furniture_mapping = {"Basic": 0.5, "Full": 1, "None": 0}
    df['Furniture state'] = furniture_mapping[input_data['furniture_state']]

    df['avg_price_area'] = get_avg_price_are(
        input_data['lat'], input_data['lon'])

    # One-hot encode legal status
    df['Legal_status_have_certificate'] = 1 if input_data['legal_status'] == 'With Certificate' else 0
    df['Legal_status_sales_contract'] = 1 if input_data['legal_status'] == 'Sales Contract' else 0

    # Rename columns to match training data
    df = df.rename(columns={
        'area': 'Area',
        'floors': 'Floors',
        'bedrooms': 'Bedrooms',
        'bathrooms': 'Bathrooms'
    })

    # Select only the features the model expects
    feature_columns = ['Area', 'Floors', 'Bedrooms', 'Bathrooms', 'Furniture state',
                       'avg_price_area', 'Legal_status_have_certificate', 'Legal_status_sales_contract']

    return df[feature_columns]


def process_input_for_pca_prediction(input_data):
    """Process input data for PCA model prediction"""
    # First get the original processed data
    df = process_input_for_prediction(input_data)

    # Scale the features first (before PCA transformation)
    scaled_features = loaded_pca_model_package['scaler'].transform(df)

    # Apply PCA transformation using the loaded PCA transformer
    pca_features = loaded_pca_model_package['pca'].transform(scaled_features)

    # Convert to DataFrame with PCA column names
    n_components = pca_features.shape[1]
    pca_columns = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(pca_features, columns=pca_columns)

    return pca_df


def get_nearest_city(location):
    lon, lat = location.longitude, location.latitude
    data = pd.DataFrame(dict(lon=lon, lat=lat), index=[0])
    transformed = transform_data(data)
    nearest_city = transformed['nearest_city'].values.squeeze()
    return nearest_city


def create_marker(m: folium.Map, location, icon_color='red', **kwargs):
    coords = [location.latitude, location.longitude]
    popup = kwargs.pop('popup', str(location))
    marker = folium.Marker(
        location=coords,
        icon=folium.Icon(color=icon_color),
        popup=popup,
        **kwargs)
    return marker


def get_markers_addresses():
    return list(map(lambda marker: marker['address'], st.session_state['markers']))


def link_two_markers(marker1, marker2, **kwargs):
    return folium.PolyLine(locations=(marker1.location, marker2.location), **kwargs)


def clear_markers():
    st.session_state['markers'] = []
    st.session_state['lines'] = []
    st.session_state['map_center'] = [14.0, 110.0]
    st.session_state['map_zoom'] = 5.5
    return folium.FeatureGroup('objects')


def create_map():
    # Define boundaries including Hoàng Sa and Trường Sa
    min_lat, min_lon = 5.5, 101.0
    max_lat, max_lon = 23.5, 117.33
    # Create map with CartoDB Voyager tiles for color
    map_vn = folium.Map(
        location=st.session_state['map_center'],
        zoom_start=st.session_state['map_zoom'],
        tiles='CartoDB Voyager',
        min_lat=min_lat,
        min_lon=min_lon,
        max_lat=max_lat,
        max_lon=max_lon,
        no_wrap=True,
        max_bounds=True
    )
    # Add markers for Hoàng Sa and Trường Sa
    folium.Marker(
        location=[16.5, 112.0],
        popup='Quần đảo Hoàng Sa',
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(map_vn)
    folium.Marker(
        location=[8.5, 113.5],
        popup='Quần đảo Trường Sa',
        icon=folium.Icon(color='green', icon='info-sign')
    ).add_to(map_vn)
    # Add caption
    return map_vn

## -----------------------------------------------------------------------------------------##
# Main Page


# Initialize session state immediately after imports
initialize_session_states()


# Title and navigation button
col_title, col_nav = st.columns([3, 1])
with col_title:
    st.title("VietNam Housing Prices Prediction")
with col_nav:
    if st.button("Phân tích dữ liệu", use_container_width=True):

        st.switch_page("pages/analysis.py")
    if st.button("Train và đánh giá model", use_container_width=True):
        st.switch_page("pages/train_and_evaluate.py")

st.markdown("""
##### A web application for predicting VietNam Housing Prices.
 
This app uses machine learning to predict the price of houses using **two different models**:

🏠 **Original Data Model (Random Forest):** Uses all original features with R² = 0.998  
📊 **PCA Model (Support Vector Regression):** Uses dimensionally reduced features with R² = 0.978

Both models take as input various features of the house such as area, floors, bedrooms, bathrooms, 
legal status, furniture state, direction, and location, and output predicted prices for comparison.
""")

# Create map after initializing session state
map_vn = create_map()
st.session_state['fg'] = folium.FeatureGroup(name="objects", control=True)

# layout and input data
col_map, col_input = st.columns([1, 1])
with col_map:
    st.header("Map of Vietnam")
    for marker_content in st.session_state["markers"]:
        st.session_state['fg'].add_child(marker_content['marker'])
    for line in st.session_state["lines"]:
        st.session_state['fg'].add_child(line)
    clean_button = st.button("Xóa điểm đánh dấu")
    if clean_button:
        st.session_state['fg'] = clear_markers()
    st_data = st_folium(map_vn, width=600, height=600,
                        feature_group_to_add=st.session_state['fg'])

with col_input:
    st.header("Enter the attributes of the housing.")
    st.markdown("""
        <style>
        div[data-testid="stNumberInput"] > div > div > input,
        div[data-testid="stSelectbox"] > div > div > select {
            width: 33% !important;
        }
        </style>
    """, unsafe_allow_html=True)

    subcol1, subcol2 = st.columns(2)
    with subcol1:
        area = st.number_input(
            "Area (square meters)",
            value=50.0,
            min_value=10.0,
            max_value=1000.0,
            step=10.0
        )
        bedrooms = st.number_input(
            "Bedrooms",
            value=2,
            min_value=1,
            max_value=20,
            step=1
        )
        bathrooms = st.number_input(
            "Bathrooms",
            value=1,
            min_value=1,
            max_value=10,
            step=1
        )
        floors = st.number_input(
            "Floors",
            value=1,
            min_value=1,
            max_value=10,
            step=1
        )

    with subcol2:
        legal_status = st.selectbox(
            "Legal Status",
            options=["With Certificate", "Sales Contract", "None"],
            index=0
        )
        furniture_state = st.selectbox(
            "Furniture State",
            options=["Basic", "Full", "None"],
            index=0
        )
        direction = st.selectbox(
            "Direction",
            options=["East", "West", "South", "North", "Northwest",
                     "Southeast", "Northeast", "Southwest"],
            index=0
        )

    address = st.text_input(
        "Address",
        placeholder="e.g., 123 Nguyễn Huệ, Quận 1, TP. Hồ Chí Minh, Việt Nam"
    )

    st.caption("Nhấn nút dưới đây để đánh dấu địa chỉ trên bản đồ.")
    locate_button = st.button("Locate")

    if address and locate_button and (not address in get_markers_addresses()):
        st.session_state['address'] = address
        location = get_location(address)
        st.session_state['location'] = location

        if location:
            # Kiểm tra nếu địa chỉ thuộc Việt Nam (bao gồm Hoàng Sa, Trường Sa)
            country = location.raw['address'].get('country', '')
            lat, lon = location.latitude, location.longitude
            is_in_vietnam = (
                ('Việt Nam' in country or 'Vietnam' in country) and
                (5.5 <= lat <= 23.5) and (101.0 <= lon <= 117.33)
            )
            if is_in_vietnam:
                housing_coords = (lat, lon)
                housing_marker = create_marker(
                    map_vn, location, icon_color='red')

                nearest_city = get_nearest_city(location)
                nearest_city_loc = get_location(nearest_city + ", Việt Nam")

                if nearest_city_loc:
                    nearest_city_coords = (
                        nearest_city_loc.latitude, nearest_city_loc.longitude)
                    distance_km = geopy.distance.distance(
                        nearest_city_coords, housing_coords).km

                    # st.session_state[
                    #     'address_output'] = f'Thành phố gần nhất: {nearest_city} | Khoảng cách: {distance_km:.2f} km'
                    # nearest_city_marker = create_marker(
                    #     map_vn, nearest_city_loc,
                    #     icon_color='green')

                    # line_markers = link_two_markers(
                    #     housing_marker, nearest_city_marker, tooltip=f'Khoảng cách: {distance_km:.2f} km')

                    st.session_state['markers'].append(
                        {'marker': housing_marker, 'address': address})
                    # st.session_state['markers'].append(
                    #     {'marker': nearest_city_marker, 'address': "n_city_" + address})
                    # st.session_state['lines'].append(line_markers)
                else:
                    st.session_state['address_output'] = f'Địa chỉ: {address} '
                    st.session_state['markers'].append(
                        {'marker': housing_marker, 'address': address})

                # Update map center and zoom
                st.session_state['map_center'] = [lat, lon]
                st.session_state['map_zoom'] = 12
                st.rerun()

                time.sleep(1)
            else:
                st.warning(
                    "Địa chỉ cần thuộc Việt Nam (bao gồm Hoàng Sa và Trường Sa). Nhập địa chỉ không thuộc Việt Nam có thể dẫn đến kết quả không chính xác.")
        else:
            st.error(
                "Không tìm thấy địa chỉ hoặc lỗi kết nối. Vui lòng thử lại với địa chỉ khác hoặc sau vài giây.")

    st.write(st.session_state['address_output'])
    button = st.button("Predict", use_container_width=True)

    if button:
        if not address in get_markers_addresses():
            st.error(
                "Bạn chưa định vị địa chỉ. Nhấn nút 'Locate' để đánh dấu trên bản đồ.")
        elif bedrooms > (area / 10):
            st.error('Lỗi: Số phòng ngủ quá lớn so với diện tích.')
        else:
            location = st.session_state['location']
            input_data = {
                "lon": location.longitude,
                "lat": location.latitude,
                "area": area,
                "floors": floors,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "legal_status": legal_status,
                "furniture_state": furniture_state,
                "direction": direction
            }

            # Process input data for both models
            st.subheader("🔄 Xử lý dữ liệu đầu vào")

            processed_df = process_input_for_prediction(input_data)
            processed_pca_df = process_input_for_pca_prediction(input_data)

            # Display processed data
            with st.expander("📋 Xem dữ liệu đã xử lý (Original Features)", expanded=False):
                st.dataframe(processed_df)

            with st.expander("📊 Xem dữ liệu đã xử lý (PCA Features)", expanded=False):
                st.dataframe(processed_pca_df)

            st.subheader("🤖 Thực hiện dự đoán")

            # Scale the features using the saved scalers and make predictions

            # Original model prediction
            scaled_features = loaded_model_package['scaler'].transform(
                processed_df)
            prediction = loaded_model_package['model'].predict(
                scaled_features).squeeze()

            # PCA model prediction (scaling already done in process_input_for_pca_prediction)
            pca_prediction = loaded_pca_model_package['model'].predict(
                processed_pca_df).squeeze()

            st.session_state['prediction'] = prediction
            st.session_state['pca_prediction'] = pca_prediction

            # Save prediction to history
            prediction_entry = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'address': address,
                'area': area,
                'floors': floors,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'legal_status': legal_status,
                'furniture_state': furniture_state,
                'direction': direction,
                'original_prediction': prediction,
                'pca_prediction': pca_prediction,
                'average_prediction': (prediction + pca_prediction) / 2
            }
            st.session_state['prediction_history'].append(prediction_entry)

            st.success("Hoàn tất dự đoán bằng cả 2 mô hình!")

    if st.session_state['prediction'] and st.session_state['pca_prediction']:
        original_pred = st.session_state['prediction']
        pca_pred = st.session_state['pca_prediction']

        st.markdown(
            """
            <style>
            [data-testid="stMetricValue"] {
                font-size: 28px;
            }
            .metric-original {
                color: #1f77b4;
            }
            .metric-pca {
                color: #ff7f0e;
            }
            .metric-average {
                color: #2ca02c;
                font-weight: bold;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Display predictions from both models
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="metric-original">',
                        unsafe_allow_html=True)
            st.metric(
                label='🏠 Mô hình dữ liệu gốc (Random Forest)',
                value=f"{original_pred:,.4f} Tỷ VND",
                help="R² = 0.998"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-pca">', unsafe_allow_html=True)
            st.metric(
                label='📊 Mô hình PCA (SVR)',
                value=f"{pca_pred:,.4f} Tỷ VND",
                help="R² = 0.978"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            avg_pred = (original_pred + pca_pred) / 2
            st.markdown('<div class="metric-average">', unsafe_allow_html=True)
            st.metric(
                label='⚖️ Giá trị trung bình',
                value=f"{avg_pred:,.4f} Tỷ VND",
                help="Trung bình của 2 mô hình"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # Show comparison analysis
        st.markdown("---")
        st.subheader("📈 Phân tích so sánh")

        diff_absolute = abs(original_pred - pca_pred)
        diff_percentage = (diff_absolute / max(original_pred, pca_pred)) * 100

        col_analysis1, col_analysis2 = st.columns(2)

        with col_analysis1:
            st.write(f"**Chênh lệch tuyệt đối:** {diff_absolute:.4f} Tỷ VND")
            st.write(f"**Chênh lệch tương đối:** {diff_percentage:.2f}%")

            if diff_percentage < 5:
                st.success("✅ Hai mô hình cho kết quả tương đồng cao")
            elif diff_percentage < 15:
                st.warning("⚠️ Có sự khác biệt nhẹ giữa hai mô hình")
            else:
                st.error("❌ Sự khác biệt lớn giữa hai mô hình")

        with col_analysis2:
            # Determine which model predicts higher
            if original_pred > pca_pred:
                st.info(
                    f"📈 Mô hình dữ liệu gốc dự đoán cao hơn {diff_absolute:.4f} Tỷ VND")
            elif pca_pred > original_pred:
                st.info(
                    f"📈 Mô hình PCA dự đoán cao hơn {diff_absolute:.4f} Tỷ VND")
            else:
                st.info("🎯 Hai mô hình dự đoán giống hệt nhau")

            # Recommendation based on model performance
            st.write("**💡 Khuyến nghị:**")
            st.write(
                "Mô hình dữ liệu gốc có độ chính xác cao hơn (R² = 0.998) so với mô hình PCA (R² = 0.978)")

    elif st.session_state['prediction']:
        # Fallback for original prediction only
        pred = st.session_state['prediction']
        st.metric(label='Giá trị nhà trung bình', value=f"{pred:,.4f} Tỷ VND")

# Prediction History Section
if st.session_state['prediction_history']:
    st.markdown("---")
    st.subheader("📋 Lịch sử dự đoán")

    # Add buttons to manage history
    col_hist1, col_hist2, col_hist3 = st.columns([1, 1, 2])

    with col_hist1:
        if st.button("🗑️ Xóa lịch sử", use_container_width=True):
            st.session_state['prediction_history'] = []
            st.rerun()

    with col_hist2:
        if st.button("📤 Xuất CSV", use_container_width=True):
            import pandas as pd
            df_history = pd.DataFrame(st.session_state['prediction_history'])
            csv = df_history.to_csv(index=False)
            st.download_button(
                label="💾 Tải xuống",
                data=csv,
                file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

    # Display history in a nice format
    with st.expander(f"📊 Xem {len(st.session_state['prediction_history'])} dự đoán gần đây", expanded=True):
        # Show last 10
        for i, entry in enumerate(reversed(st.session_state['prediction_history'][-10:])):
            with st.container():
                st.markdown(
                    f"**#{len(st.session_state['prediction_history']) - i}** - {entry['timestamp']}")

                col_h1, col_h2, col_h3 = st.columns([2, 1, 1])

                with col_h1:
                    st.write(f"📍 **Địa chỉ:** {entry['address']}")
                    st.write(
                        f"🏠 **Thông tin:** {entry['area']}m² | {entry['bedrooms']} phòng ngủ | {entry['bathrooms']} phòng tắm | {entry['floors']} tầng")
                    st.write(
                        f"📋 **Tình trạng:** {entry['legal_status']} | {entry['furniture_state']} | {entry['direction']}")

                with col_h2:
                    st.metric("🏠 Mô hình gốc",
                              f"{entry['original_prediction']:,.4f} Tỷ")
                    st.metric("📊 Mô hình PCA",
                              f"{entry['pca_prediction']:,.4f} Tỷ")

                with col_h3:
                    st.metric("⚖️ Trung bình",
                              f"{entry['average_prediction']:,.4f} Tỷ")
                    diff = abs(entry['original_prediction'] -
                               entry['pca_prediction'])
                    diff_pct = (
                        diff / max(entry['original_prediction'], entry['pca_prediction'])) * 100
                    st.write(f"📈 Chênh lệch: {diff_pct:.1f}%")

                st.markdown("---")

        if len(st.session_state['prediction_history']) > 10:
            st.info(
                f"Hiển thị 10 dự đoán gần nhất. Tổng cộng có {len(st.session_state['prediction_history'])} dự đoán.")

    # Statistics
    if len(st.session_state['prediction_history']) > 1:
        with st.expander("📊 Thống kê tổng quan", expanded=False):
            df_stats = pd.DataFrame(st.session_state['prediction_history'])

            col_s1, col_s2, col_s3 = st.columns(3)

            with col_s1:
                st.metric("Tổng số dự đoán", len(df_stats))
                st.metric("Giá trung bình (Mô hình gốc)",
                          f"{df_stats['original_prediction'].mean():,.4f} Tỷ")

            with col_s2:
                st.metric("Giá cao nhất",
                          f"{df_stats['average_prediction'].max():,.4f} Tỷ")
                st.metric("Giá thấp nhất",
                          f"{df_stats['average_prediction'].min():,.4f} Tỷ")

            with col_s3:
                avg_diff = abs(
                    df_stats['original_prediction'] - df_stats['pca_prediction']).mean()
                st.metric("Chênh lệch TB giữa 2 mô hình",
                          f"{avg_diff:,.4f} Tỷ")
                avg_area = df_stats['area'].mean()
                st.metric("Diện tích trung bình", f"{avg_area:.1f} m²")
