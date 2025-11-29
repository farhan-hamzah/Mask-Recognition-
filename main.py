import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ==========================================
# KONFIGURASI
# ==========================================
MODEL_PATH = 'mask_detector_model.keras'  # Pastikan nama file sesuai
THRESHOLD = 0.5  # Batas ambang (0.5 ke atas = Masker)

# ==========================================
# 1. LOAD MODEL & FACE DETECTOR
# ==========================================
print("Loading model...")
model = load_model(MODEL_PATH)

# Kita pakai detektor wajah bawaan OpenCV (Haar Cascade) agar ringan
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ==========================================
# 2. BUKA WEBCAM
# ==========================================
cap = cv2.VideoCapture(0) # 0 biasanya ID untuk webcam laptop default

if not cap.isOpened():
    print("Error: Tidak bisa membuka kamera.")
    exit()

print("Kamera dimulai. Tekan 'q' untuk keluar.")

while True:
    # Baca frame dari kamera
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame biar seperti cermin
    frame = cv2.flip(frame, 1)

    # Deteksi wajah butuh gambar grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Deteksi wajah-wajah di frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    # Loop setiap wajah yang ditemukan
    for (x, y, w, h) in faces:
        # --- PREPROCESSING ---
        # Ambil bagian wajah saja (ROI - Region of Interest)
        face_img = frame[y:y+h, x:x+w]
        
        # Pastikan wajah ada isinya (mencegah error)
        if face_img.shape[0] == 0 or face_img.shape[1] == 0:
            continue

        # Ubah warna ke RGB (Model dilatih dengan RGB)
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Resize ke 224x224 (Sesuai input model MobileNetV2)
        face_img = cv2.resize(face_img, (224, 224))
        
        # Normalisasi (ubah pixel jadi 0-1)
        face_img = face_img.astype('float32') / 255.0
        
        # Tambah dimensi batch jadi (1, 224, 224, 3)
        face_input = np.expand_dims(face_img, axis=0)

        # --- PREDIKSI ---
        prediction = model.predict(face_input, verbose=0)[0][0]

        # --- LOGIKA LABEL ---
        # Berdasarkan training: 1 = Mask, 0 = No Mask
        if prediction > THRESHOLD:
            label = "Masker"
            color = (0, 255, 0) # Hijau
            confidence = prediction
        else:
            label = "Tanpa Masker"
            color = (0, 0, 255) # Merah
            confidence = 1 - prediction

        # Format teks persentase
        label_text = f"{label} ({confidence*100:.1f}%)"

        # --- VISUALISASI ---
        # Gambar kotak di wajah
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Taruh label di atas kotak
        cv2.putText(frame, label_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Tampilkan window
    cv2.imshow('Mask Detection Real-Time', frame)

    # Tekan 'q' untuk stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan
cap.release()
cv2.destroyAllWindows()