import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import joblib
from tensorflow.keras.models import load_model
import os

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Harga Saham Microsoft",
    page_icon="📈",
    layout="wide"
)

# Konstanta
TICKER = 'MSFT'
SEQUENCE_LENGTH = 60  # Harus sama dengan yang digunakan saat training
MODEL_PATH = os.path.join('artifacts', 'best_msft_lstm_model.keras')
SCALER_PATH = os.path.join('artifacts', 'scaler.pkl')

# Header aplikasi
st.title("📈 Prediksi Harga Saham Microsoft (MSFT)")
st.markdown("""
Aplikasi ini menggunakan model LSTM (Long Short-Term Memory) untuk memprediksi harga saham Microsoft (MSFT).
Model telah dilatih menggunakan data historis dari Yahoo Finance.
""")

# Sidebar untuk parameter
st.sidebar.header("Parameter")

# Pilih tanggal
today = datetime.now()
default_start_date = today - timedelta(days=365)  # 1 tahun yang lalu
start_date = st.sidebar.date_input("Tanggal Mulai Data Historis", default_start_date)
end_date = st.sidebar.date_input("Tanggal Akhir Data Historis", today)

# Fungsi untuk mengunduh data saham
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            st.error(f"Tidak ada data yang ditemukan untuk {ticker}")
            return None
        return data
    except Exception as e:
        st.error(f"Error saat mengunduh data: {e}")
        return None

# Mengunduh data saham
with st.spinner("Mengunduh data saham terbaru..."):
    data = fetch_stock_data(TICKER, start_date, end_date)

if data is None or data.empty:
    st.error("Tidak dapat mengunduh data saham. Periksa koneksi internet Anda.")
    st.stop()

# Periksa dan tangani MultiIndex columns jika ada
if isinstance(data.columns, pd.MultiIndex):
    # Flatten kolom multi-index menjadi kolom tunggal
    data.columns = [col[0] for col in data.columns]
    st.info("Data dengan struktur kolom multi-level telah dikonversi untuk visualisasi.")

# Menampilkan data historis
st.subheader("Data Historis Harga Saham Microsoft")
col1, col2 = st.columns([3, 1])

with col1:
    # Plot data harga penutupan dengan Plotly
    fig = px.line(data, x=data.index, y='Close', title='Harga Penutupan Saham MSFT')
    fig.update_layout(xaxis_title='Tanggal', yaxis_title='Harga (USD)')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.dataframe(data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10), height=300)
    st.caption("10 data terakhir")

# Prediksi harga saham (jika model tersedia)
st.subheader("Prediksi Harga Saham")

# Cek apakah model dan scaler tersedia
model_exists = os.path.exists(MODEL_PATH)
scaler_exists = os.path.exists(SCALER_PATH)

if not model_exists or not scaler_exists:
    st.warning("Model prediksi belum tersedia. Jalankan 'train_model.py' terlebih dahulu untuk melatih model.")
else:
    try:
        # Load model dan scaler
        with st.spinner("Memuat model..."):
            model = load_model(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            
        # Membuat sekuens data untuk prediksi
        def create_sequence(data, seq_length):
            if len(data) <= seq_length:
                st.error(f"Tidak cukup data untuk membuat sekuens (butuh minimal {seq_length+1} data)")
                return None
            return np.array(data[-seq_length:]).reshape(1, seq_length, 1)
        
        # Memproses data untuk prediksi
        close_data = data['Close'].values.reshape(-1, 1)
        scaled_data = scaler.transform(close_data)
        
        # Jika data cukup, buat sekuens dan lakukan prediksi
        if len(scaled_data) > SEQUENCE_LENGTH:
            # Ambil data terakhir untuk prediksi hari berikutnya
            last_sequence = scaled_data[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, 1)
            
            # Prediksi
            with st.spinner("Membuat prediksi..."):
                next_day_prediction_scaled = model.predict(last_sequence)
                next_day_prediction = scaler.inverse_transform(next_day_prediction_scaled)[0][0]
            
            # Tampilkan prediksi
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label=f"Prediksi Harga MSFT untuk Hari Berikutnya", 
                    value=f"${next_day_prediction:.2f}",
                    delta=f"{next_day_prediction - data['Close'].iloc[-1]:.2f} USD"
                )
            
            # Prediksi beberapa hari ke depan
            future_days = 7  # Jumlah hari untuk diprediksi
            
            with col2:
                future_prediction_btn = st.button("Tampilkan Prediksi 7 Hari Ke Depan")
            
            if future_prediction_btn:
                with st.spinner("Memprediksi harga untuk 7 hari ke depan..."):
                    # Fungsi untuk memprediksi beberapa hari ke depan
                    def predict_future(model, scaler, last_sequence, days):
                        predictions = []
                        current_sequence = last_sequence.copy()
                        
                        for _ in range(days):
                            # Prediksi 1 hari ke depan
                            pred_scaled = model.predict(current_sequence)
                            pred = scaler.inverse_transform(pred_scaled)[0][0]
                            predictions.append(pred)
                            
                            # Update sequence untuk prediksi berikutnya
                            current_sequence = np.append(current_sequence[:,1:,:], pred_scaled.reshape(1,1,1), axis=1)
                        
                        return predictions
                    
                    future_predictions = predict_future(model, scaler, last_sequence, future_days)
                    
                    # Buat dataframe untuk hasil prediksi
                    last_date = data.index[-1]
                    future_dates = [last_date + timedelta(days=i+1) for i in range(future_days)]
                    future_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted_Close': future_predictions
                    })
                    
                    # Tampilkan tabel prediksi
                    st.subheader("Prediksi Harga 7 Hari Ke Depan")
                    st.dataframe(future_df.set_index('Date'))
                    
                    # Plot prediksi
                    fig = go.Figure()
                    
                    # Data historis
                    fig.add_trace(go.Scatter(
                        x=data.index[-30:], 
                        y=data['Close'][-30:], 
                        mode='lines',
                        name='Harga Aktual',
                        line=dict(color='blue')
                    ))
                    
                    # Data prediksi
                    fig.add_trace(go.Scatter(
                        x=future_df['Date'], 
                        y=future_df['Predicted_Close'],
                        mode='lines+markers',
                        name='Prediksi',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig.update_layout(
                        title='Prediksi Harga Saham MSFT',
                        xaxis_title='Tanggal',
                        yaxis_title='Harga (USD)',
                        legend=dict(y=0.99, x=0.01),
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Tidak cukup data historis untuk membuat prediksi. Minimum: {SEQUENCE_LENGTH} hari.")
    except Exception as e:
        st.error(f"Error saat membuat prediksi: {e}")

# Informasi tambahan
st.subheader("Tentang Model")
st.markdown("""
Model LSTM (Long Short-Term Memory) digunakan untuk prediksi ini dengan konfigurasi:
- Sequence Length: 60 hari
- 3 layer LSTM Bidirectional
- Dropout layers untuk menghindari overfitting
- Model dilatih menggunakan Mean Squared Error (MSE)

**Catatan Penting:**
- Prediksi pasar saham memiliki ketidakpastian tinggi dan dipengaruhi banyak faktor
- Harap gunakan prediksi ini sebagai salah satu referensi saja, bukan keputusan investasi utama
- Model hanya mempertimbangkan data historis harga, bukan berita atau fundamental
""")