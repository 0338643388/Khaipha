import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates

# --- 1. CẤU HÌNH TRANG  ---
st.set_page_config(page_title="Hệ thống dự báo chứng khoán", layout="wide")

# --- HÀM BỔ TRỢ ---
def create_features(data):
    df = data.copy()
    df['lp'] = np.log(df['close'])
    df['Daily_Return'] = df['close'].pct_change()
    for w in [5, 10, 20]:
        df[f'MA_{w}'] = df['close'].rolling(window=w).mean()
        df[f'Volatility_{w if w < 20 else 10}'] = df['close'].rolling(window=w if w < 20 else 10).std()
    df['Momentum_5'] = df['close'].diff(5)
    df['Momentum_10'] = df['close'].diff(10)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    return df

# --- GIAO DIỆN CHÍNH ---
st.title("📈 Stock Prediction Webapp")
st.markdown("---")

# Sidebar
stocks = {"HPG": "Hòa Phát", "VCB": "Vietcombank", "VNM": "Vinamilk", "FPT": "FPT Corporation"}
ticker = st.sidebar.selectbox("Chọn mã cổ phiếu:", list(stocks.keys()))
n_days = st.sidebar.slider("Số ngày dự báo đệ quy:", 1, 100, 1)
model_choice = st.sidebar.selectbox("Model sử dụng", ["Random Forest", "Linear Regression", "Gradient Boosting"])

# Load dữ liệu 
try:
    df = pd.read_csv('HPG_stock_price.csv', parse_dates=['time'], index_col='time')
    model_metrics = pd.read_csv('model_performance_comparison.csv')
except Exception as e:
    st.error(f"Thiếu file dữ liệu: {e}")
    st.stop()

if df is not None:
    # --- PHẦN 1: TỔNG QUAN LỊCH SỬ ---
    st.write(f"Dữ liệu phân tích từ **{df.index[0].date()}** đến **{df.index[-1].date()}**")
    
    fig_hist, ax1 = plt.subplots(figsize=(16, 5))
    ax1.set_ylabel('Giá đóng cửa (VND)', color='blue', fontweight='bold')
    ax1.plot(df.index, df['close'], color='blue', linewidth=1.5, label='Giá đóng cửa')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.2)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Khối lượng', color='tab:blue', fontweight='bold')
    ax2.bar(df.index, df['volume'], color='tab:blue', alpha=0.2, label='Khối lượng')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    plt.title(f'Xu hướng Giá và Khối lượng giao dịch {ticker}', fontsize=14, fontweight='bold')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    st.pyplot(fig_hist)

    df_processed = create_features(df)
    st.markdown("---")

    # --- PHẦN 2: CHIA CỘT KẾT QUẢ VÀ ĐỀ XUẤT ---
    if st.button("DỰ BÁO PHIÊN TIẾP THEO"):
        with st.spinner('Đang phân tích dữ liệu...'):
            try:
                # 1. Logic Dự báo
                clean_name = model_choice.replace(" ", "_").lower()
                model = joblib.load(f'{clean_name}_model.pkl')
                scaler = joblib.load(f'scaler.pkl')
                
                selected_row = model_metrics[model_metrics['Model'] == model_choice]
                mae_val = selected_row['Test_MAE'].values[0] if not selected_row.empty else 0
                
                features = ['volume', 'Daily_Return', 'MA_5', 'MA_10', 'MA_20', 
                            'Momentum_5', 'Momentum_10', 'Volatility_5', 'Volatility_10', 'RSI', 'MACD']
                lags = [1, 2, 3, 5, 10, 20]
                
                last_idx = len(df_processed) - 1
                data_values = df_processed[features].values
                X_next = np.array([data_values[last_idx + 1 - lag] for lag in lags]).flatten().reshape(1, -1)
                
                X_next_scaled = scaler.transform(X_next)
                pred_ret = model.predict(X_next_scaled)[0]
                
                last_close = df_processed.iloc[-1]['close']
                predicted_price = last_close * np.exp(pred_ret)
                change_pct = (np.exp(pred_ret) - 1) * 100

                # --- CHIA CỘT ---
                col_left, col_right = st.columns([2, 1])

                with col_left:
                    st.subheader("Kết quả phân tích kỹ thuật")
                    
                    # Hiển thị Metric
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Giá hiện tại", f"{last_close*1000:,.0f}")
                    m2.metric("Giá dự báo", f"{predicted_price*1000:,.0f}", f"{change_pct:+.2f}%")
                    m3.metric("Sai số TB (MAE)", f"{mae_val:,.0f}")

                    # Biểu đồ Zoom 30 ngày
                    fig_pred, ax = plt.subplots(figsize=(10, 6))
                    recent_df = df.tail(30)
                    ax.plot(recent_df.index, recent_df['close'], marker='o', label='Thực tế', color='#1f77b4')
                    
                    next_date = df.index[-1] + pd.Timedelta(days=1)
                    ax.scatter(next_date, predicted_price, color='red', s=120, label='Dự báo', zorder=5)
                    ax.plot([df.index[-1], next_date], [last_close, predicted_price], color='red', linestyle='--', alpha=0.6)
                    
                    ax.set_title(f"Vùng biến động dự báo ({model_choice})")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig_pred)

                with col_right:
                    st.subheader("Đề xuất hành động")
                    
                    # 1. Trạng thái Indicators
                    rsi_now = df_processed['RSI'].iloc[-1]
                    macd_now = df_processed['MACD'].iloc[-1]
                    
                    st.info(f"**Chỉ số RSI hiện tại: {rsi_now:.2f}**")
                    if rsi_now > 70: st.warning("Thị trường đang ở vùng QUÁ MUA. Cân nhắc chốt lời.")
                    elif rsi_now < 30: st.success("Thị trường đang ở vùng QUÁ BÁN. Cơ hội tích lũy.")
                    else: st.write("RSI đang ở vùng trung tính.")

                    st.markdown("---")

                    # 2. Khuyến nghị dựa trên mô hình
                    st.write("**Chiến lược từ Model:**")
                    if change_pct > 1.5:
                        st.success("### MUA MẠNH")
                        st.write("Mô hình dự báo lực tăng mạnh. Có thể gia tăng tỷ trọng.")
                    elif change_pct > 0.3:
                        st.info("### MUA / NẮM GIỮ")
                        st.write("Dự báo tăng nhẹ. Phù hợp nắm giữ quan sát thêm.")
                    elif change_pct < -1.5:
                        st.error("### BÁN / ĐỨNG NGOÀI")
                        st.write("Cảnh báo rủi ro giảm sâu. Cân nhắc hạ tỷ trọng.")
                    else:
                        st.warning("### ĐI NGANG (SIDEWAY)")
                        st.write("Biến động không rõ ràng. Nên kiên nhẫn quan sát.")

                    st.markdown("---")
                    
                    # 3. Phân tích bổ sung
                    st.write("**Lưu ý rủi ro:**")
                    st.caption(f"Dự báo dựa trên dữ liệu quá khứ. Sai số trung bình hiện tại của mô hình là {mae_val:,.0f} VNĐ. Hãy kết hợp với tin tức vĩ mô.")

            except Exception as e:
                st.error(f"Lỗi xử lý: {e}")
else:
    st.info("Hãy chọn các thông số ở Sidebar và nhấn nút Dự báo.")