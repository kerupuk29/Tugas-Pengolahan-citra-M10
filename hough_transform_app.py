import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Konfigurasi Halaman
st.set_page_config(page_title="Hough Transform: Garis & Lingkaran", layout="wide")
st.title("🔍 Transformasi Hough: Deteksi Garis & Lingkaran")
st.markdown("""
Aplikasi ini mendeteksi **Garis** dan **Lingkaran** pada citra menggunakan Transformasi Hough.
Silakan upload gambar yang memiliki bentuk geometris jelas (seperti koin, jalan raya, atau bentuk dasar).
""")

# --- Sidebar Pengaturan ---
st.sidebar.header("1. Upload Gambar")
uploaded_file = st.sidebar.file_uploader("Pilih gambar (JPG/PNG)", type=["jpg", "png", "jpeg"])

st.sidebar.header("2. Metode Deteksi")
detection_mode = st.sidebar.radio("Pilih apa yang ingin dideteksi:", ["Deteksi Garis (Hough Line)", "Deteksi Lingkaran (Hough Circle)"])

# --- Panel Utama ---
if uploaded_file is not None:
    # Baca Gambar
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Konversi ke BGR untuk OpenCV jika perlu (Streamlit/PIL pakai RGB)
    if len(img_array.shape) == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        gray_img = img_array

    # Tampilkan Gambar Asli
    st.subheader("Gambar Asli")
    st.image(image, caption="Input Image", use_container_width=True)

    # --- Pre-processing (Canny Edge Detection) ---
    # Hough Transform butuh citra tepi (edge map) dulu
    st.sidebar.markdown("---")
    st.sidebar.header("3. Pengaturan Canny (Tepi)")
    canny_min = st.sidebar.slider("Canny Min Threshold", 0, 255, 50)
    canny_max = st.sidebar.slider("Canny Max Threshold", 0, 255, 150)
    
    edges = cv2.Canny(gray_img, canny_min, canny_max)

    # --- Logika Deteksi ---
    
    if detection_mode == "Deteksi Garis (Hough Line)":
        st.sidebar.markdown("---")
        st.sidebar.header("4. Parameter Hough Line")
        
        # Parameter HoughLinesP
        rho = st.sidebar.slider("Rho (Resolusi Jarak)", 1, 10, 1)
        theta = np.pi / 180  # Biasanya tetap 1 derajat
        threshold = st.sidebar.slider("Threshold (Min. intersection)", 10, 200, 50)
        min_line_len = st.sidebar.slider("Min Line Length", 0, 200, 50)
        max_line_gap = st.sidebar.slider("Max Line Gap", 0, 100, 10)
        
        # Proses Deteksi
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=min_line_len, maxLineGap=max_line_gap)
        
        # Gambar garis di atas citra asli
        result_img = img_array.copy()
        line_count = 0
        if lines is not None:
            line_count = len(lines)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result_img, (x1, y1), (x2, y2), (0, 255, 0), 3) # Warna Hijau
        
        # Tampilkan Hasil
        col1, col2 = st.columns(2)
        with col1:
            st.image(edges, caption="Citra Tepi (Canny)", use_container_width=True)
        with col2:
            st.image(result_img, caption=f"Hasil Deteksi Garis (Ditemukan: {line_count})", use_container_width=True)
            
        st.success(f"Berhasil mendeteksi **{line_count}** garis.")

    elif detection_mode == "Deteksi Lingkaran (Hough Circle)":
        st.sidebar.markdown("---")
        st.sidebar.header("4. Parameter Hough Circle")
        
        # HoughCircle bekerja lebih baik dengan blur untuk mengurangi noise
        blur_ksize = st.sidebar.slider("Gaussian Blur Kernel (Ganjil)", 1, 15, 9, step=2)
        gray_blurred = cv2.GaussianBlur(gray_img, (blur_ksize, blur_ksize), 2)
        
        # Parameter HoughCircles
        # dp: Inverse ratio resolusi (1 = sama dengan input, 2 = setengahnya)
        dp = st.sidebar.slider("DP (Resolusi Akumulator)", 1.0, 5.0, 1.2)
        # minDist: Jarak minimum antar pusat lingkaran
        min_dist = st.sidebar.slider("Min Distance antar Pusat", 10, 200, 50)
        # param1: Threshold tinggi untuk Canny (internal function)
        param1 = st.sidebar.slider("Param 1 (Canny High)", 10, 300, 50)
        # param2: Threshold akumulator (semakin kecil, semakin banyak lingkaran palsu)
        param2 = st.sidebar.slider("Param 2 (Threshold Akumulator)", 10, 100, 30)
        
        min_radius = st.sidebar.slider("Min Radius", 0, 200, 0)
        max_radius = st.sidebar.slider("Max Radius", 0, 500, 0) # 0 artinya tidak terbatas
        
        # Proses Deteksi
        circles = cv2.HoughCircles(gray_blurred, cv2.HoughCircles, dp, min_dist,
                                   param1=param1, param2=param2, 
                                   minRadius=min_radius, maxRadius=max_radius)
        
        # Gambar lingkaran
        result_img = img_array.copy()
        circle_count = 0
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            circle_count = len(circles[0, :])
            for i in circles[0, :]:
                # Gambar lingkaran luar (Hijau)
                cv2.circle(result_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # Gambar titik pusat (Merah)
                cv2.circle(result_img, (i[0], i[1]), 2, (255, 0, 0), 3)

        # Tampilkan Hasil
        col1, col2 = st.columns(2)
        with col1:
            st.image(edges, caption="Citra Tepi (Canny - Referensi)", use_container_width=True)
        with col2:
            st.image(result_img, caption=f"Hasil Deteksi Lingkaran (Ditemukan: {circle_count})", use_container_width=True)
            
        st.success(f"Berhasil mendeteksi **{circle_count}** lingkaran.")

    # Tombol Download Hasil
    st.markdown("---")
    result_pil = Image.fromarray(result_img)
    buf = io.BytesIO()
    result_pil.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    
    st.download_button(
        label="⬇️ Download Hasil Deteksi (JPG)",
        data=byte_im,
        file_name="hasil_hough.jpg",
        mime="image/jpeg"
    )

else:
    st.info("Silakan upload gambar melalui sidebar sebelah kiri untuk memulai.")
