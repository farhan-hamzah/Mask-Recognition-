"""
MASK DETECTION TRAINING SCRIPT
Dataset: Face Mask Detection (Kaggle)
Author: Computer Vision Training Script
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from glob import glob
import random

# ============================================
# FUNGSI 1: PARSING XML ANNOTATIONS
# ============================================

def parse_xml_annotation(xml_file, images_dir):
    """
    Parse file XML (Pascal VOC format) dari Kaggle dataset
    
    Args:
        xml_file: Path ke file XML
        images_dir: Path ke folder images
    
    Returns:
        List of dict berisi info setiap face (image, label, box)
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Extract image filename
    filename = root.find('filename').text
    image_path = os.path.join(images_dir, filename)
    
    # Check apakah file image ada
    if not os.path.exists(image_path):
        return []
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return []
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Extract semua objects (faces)
    faces_data = []
    
    for obj in root.findall('object'):
        label = obj.find('name').text
        
        # Extract bounding box
        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        
        # Crop face region
        face = image[ymin:ymax, xmin:xmax]
        
        # Skip jika crop terlalu kecil
        if face.shape[0] < 20 or face.shape[1] < 20:
            continue
        
        # Simplifikasi label: with_mask=1, others=0
        if label == 'with_mask':
            simplified_label = 1
        else:
            simplified_label = 0
        
        faces_data.append({
            'image': face,
            'label': simplified_label,
            'original_label': label
        })
    
    return faces_data


# ============================================
# FUNGSI 2: LOAD DATASET
# ============================================
def load_kaggle_dataset(dataset_path):
    """
    Load dataset dengan UNDERSAMPLING (Memotong data mayoritas)
    agar seimbang 50:50
    """
    annotations_dir = os.path.join(dataset_path, 'annotations')
    images_dir = os.path.join(dataset_path, 'images')
    
    print(f"Loading dataset from: {dataset_path}")
    
    # Tampung data sementara
    mask_data = []
    no_mask_data = []
    
    # Get all XML files
    xml_files = glob(os.path.join(annotations_dir, '*.xml'))
    print(f"Found {len(xml_files)} XML files")
    
    # Parse each XML file
    for i, xml_file in enumerate(xml_files):
        if (i + 1) % 50 == 0:
            print(f"Processing {i+1}/{len(xml_files)} files...")
        
        faces = parse_xml_annotation(xml_file, images_dir)
        
        # Pisahkan langsung ke keranjang masing-masing
        for face in faces:
            if face['label'] == 1:
                mask_data.append(face)
            else:
                no_mask_data.append(face)
    
    print(f"\nOriginal Data -> With Mask: {len(mask_data)}, Without Mask: {len(no_mask_data)}")
    
    # --- PROSES PEMOTONGAN DATA (UNDERSAMPLING) ---
    # Kita potong jumlah data Masker agar sama persis dengan Tanpa Masker
    min_len = len(no_mask_data)
    
    # Ambil data masker secara acak sejumlah min_len
    if len(mask_data) > min_len:
        print(f"⚠️ Melakukan Undersampling: Memotong data Masker dari {len(mask_data)} menjadi {min_len}...")
        random.shuffle(mask_data) # Acak dulu
        mask_data = mask_data[:min_len] # Potong
    
    # Gabungkan kembali
    all_data = mask_data + no_mask_data
    random.shuffle(all_data) # Acak urutannya biar tercampur
    
    print(f"Balanced Data -> With Mask: {len(mask_data)}, Without Mask: {len(no_mask_data)}")
    print(f"Total Training Data: {len(all_data)}")
    
    return all_data

# ============================================
# FUNGSI 3: PREPROCESSING
# ============================================

def preprocess_dataset(dataset, target_size=(224, 224)):
    """
    Resize dan normalize images
    
    Args:
        dataset: List of dict
        target_size: Target image size
    
    Returns:
        X: numpy array of images
        y: numpy array of labels
    """
    print("\nPreprocessing images...")
    
    images = []
    labels = []
    
    for data in dataset:
        # Resize image
        image = cv2.resize(data['image'], target_size)
        
        # Normalize pixel values [0-255] → [0-1]
        image = image.astype('float32') / 255.0
        
        images.append(image)
        labels.append(data['label'])
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"With Mask: {np.sum(y == 1)}, Without Mask: {np.sum(y == 0)}")
    
    return X, y


# ============================================
# FUNGSI 4: CREATE DATA GENERATORS
# ============================================

def create_data_generators(X_train, y_train, X_val, y_val, batch_size=32):
    """
    Create data generators dengan augmentation
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        batch_size: Batch size
    
    Returns:
        train_generator, val_generator
    """
    # Data augmentation untuk training
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Tidak ada augmentation untuk validation
    val_datagen = ImageDataGenerator()
    
    # Create generators
    train_generator = train_datagen.flow(
        X_train, y_train,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_generator = val_datagen.flow(
        X_val, y_val,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_generator, val_generator


# ============================================
# FUNGSI 4B: CALCULATE CLASS WEIGHTS
# ============================================

def calculate_class_weights(y_train):
    """
    Calculate class weights untuk mengatasi imbalance
    
    Args:
        y_train: Training labels
    
    Returns:
        class_weight: Dictionary of class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    # Hitung class weights
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight = dict(zip(classes, weights))
    
    print(f"\nClass weights calculated:")
    print(f"  Class 0 (No Mask): {class_weight[0]:.2f}")
    print(f"  Class 1 (Mask):    {class_weight[1]:.2f}")
    
    return class_weight


# ============================================
# FUNGSI 5: BUILD MODEL (TRANSFER LEARNING)
# ============================================

def build_model(input_shape=(224, 224, 3)):
    """
    Build model menggunakan MobileNetV2 (Transfer Learning)
    
    Args:
        input_shape: Shape input image
    
    Returns:
        Compiled model
    """
    print("\nBuilding model with Transfer Learning (MobileNetV2)...")
    
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Build model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    return model


# ============================================
# FUNGSI 6: BUILD CUSTOM CNN (ALTERNATIF)
# ============================================

def build_custom_cnn(input_shape=(224, 224, 3)):
    """
    Build CNN dari scratch (lebih ringan)
    
    Args:
        input_shape: Shape input image
    
    Returns:
        Compiled model
    """
    print("\nBuilding Custom CNN model...")
    
    model = Sequential([
        # Block 1
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Block 2
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Block 3
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Block 4
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Classifier
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    return model


# ============================================
# FUNGSI 7: SETUP CALLBACKS
# ============================================

def setup_callbacks(model_save_path='best_mask_detector.h5'):
    """
    Setup callbacks untuk training
    
    Args:
        model_save_path: Path untuk save model
    
    Returns:
        List of callbacks
    """
    # Early Stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Model Checkpoint
    checkpoint = ModelCheckpoint(
        model_save_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Reduce Learning Rate
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    return [early_stop, checkpoint, reduce_lr]


# ============================================
# FUNGSI 8: TRAIN MODEL
# ============================================

def train_model(model, train_gen, val_gen, callbacks, epochs=50, class_weights=None):
    """
    Train model
    
    Args:
        model: Model yang akan ditraining
        train_gen: Training generator
        val_gen: Validation generator
        callbacks: List of callbacks
        epochs: Number of epochs
        class_weights: Dictionary of class weights untuk imbalance
    
    Returns:
        model, history
    """
    print("\n" + "="*50)
    print("STARTING TRAINING")
    print("="*50)
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,  # Tambahkan class weights
        verbose=1
    )
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED!")
    print("="*50)
    
    return model, history


# ============================================
# FUNGSI 9: FINE-TUNING (OPTIONAL)
# ============================================

def fine_tune_model(model, train_gen, val_gen, callbacks, epochs=20):
    """
    Fine-tune model dengan unfreeze beberapa layers
    
    Args:
        model: Trained model
        train_gen: Training generator
        val_gen: Validation generator
        callbacks: List of callbacks
        epochs: Number of epochs
    
    Returns:
        model, history
    """
    print("\n" + "="*50)
    print("STARTING FINE-TUNING")
    print("="*50)
    
    # Unfreeze base model
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Freeze semua kecuali top 20 layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    # Recompile dengan learning rate lebih kecil
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "="*50)
    print("FINE-TUNING COMPLETED!")
    print("="*50)
    
    return model, history


# ============================================
# FUNGSI 10: EVALUASI MODEL
# ============================================

def evaluate_model(model, X_test, y_test):
    """
    Evaluasi performa model
    
    Args:
        model: Trained model
        X_test: Test images
        y_test: Test labels
    
    Returns:
        accuracy, precision, recall, f1
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Predict
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nAccuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['No Mask', 'Mask']))
    
    return accuracy, precision, recall, f1


# ============================================
# FUNGSI 11: PLOT TRAINING HISTORY
# ============================================

def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training & validation metrics
    
    Args:
        history: History dari model.fit
        save_path: Path untuk save plot
    """
    plt.figure(figsize=(12, 4))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nTraining history plot saved as '{save_path}'")
    plt.show()


# ============================================
# MAIN TRAINING PIPELINE
# ============================================

def main_training_pipeline(dataset_path, use_custom_cnn=False):
    """
    Complete training pipeline
    
    Args:
        dataset_path: Path ke dataset Kaggle
        use_custom_cnn: Jika True, gunakan custom CNN. Jika False, gunakan Transfer Learning
    """
    # Configuration
    TARGET_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 30
    FINE_TUNE_EPOCHS = 20
    MODEL_SAVE_PATH = 'mask_detector_model.keras'
    
    print("="*50)
    print("MASK DETECTION TRAINING PIPELINE")
    print("="*50)
    
    # Step 1: Load dataset
    print("\n[1/9] Loading Kaggle dataset...")
    dataset = load_kaggle_dataset(dataset_path)
    
    if len(dataset) == 0:
        print("ERROR: No data loaded! Check your dataset path.")
        return None
    
    # Step 2: Preprocess
    print("\n[2/9] Preprocessing data...")
    X, y = preprocess_dataset(dataset, target_size=TARGET_SIZE)
    
    # Step 3: Split dataset
    print("\n[3/9] Splitting dataset...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Train: {X_train.shape[0]} samples")
    print(f"Val:   {X_val.shape[0]} samples")
    print(f"Test:  {X_test.shape[0]} samples")
    
    # Step 4: Create data generators
    print("\n[4/9] Creating data generators...")
    train_gen, val_gen = create_data_generators(
        X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE
    )
    
    # Step 4B: Calculate class weights untuk mengatasi imbalance
    print("\n[4B/9] Calculating class weights...")
    class_weights = {0: 4.0, 1: 1.0}
    print(f"Weights set to: {class_weights}")
    #class_weights = calculate_class_weights(y_train)
    
    # Step 5: Build model
    print("\n[5/9] Building model...")
    if use_custom_cnn:
        model = build_custom_cnn(input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))
    else:
        model = build_model(input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))
    
    # Step 6: Setup callbacks
    print("\n[6/9] Setting up callbacks...")
    callbacks = setup_callbacks(model_save_path=MODEL_SAVE_PATH)
    
    # Step 7: Train model
    print("\n[7/9] Training model...")
    model, history = train_model(
        model, train_gen, val_gen, callbacks, epochs=EPOCHS, 
        class_weights=class_weights  # Pass class weights
    )
    # Step 7.5: Fine-tuning
    print("\n[7.5/9] Starting Fine-Tuning (Meningkatkan Kecerdasan)...")
    # Kita panggil fungsi fine_tune yang sudah ada di kodemu
    model, history_fine = fine_tune_model(
        model, 
        train_gen, 
        val_gen, 
        callbacks, 
        epochs=20 # Tambah 20 epoch lagi khusus untuk pendalaman materi
    )
    # Step 8: Evaluate
    print("\n[8/9] Evaluating model...")
    evaluate_model(model, X_test, y_test)
    
    # Step 9: Plot history
    print("\n[9/9] Plotting training history...")
    plot_training_history(history)
    
    print("\n" + "="*50)
    print("TRAINING PIPELINE COMPLETED!")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print("="*50)
    
    return model


# ============================================
# TEST SINGLE PREDICTION
# ============================================

def test_single_prediction(model_path, image_path):
    """
    Test prediksi pada single image
    
    Args:
        model_path: Path ke saved model
        image_path: Path ke test image
    """
    # Load model
    model = keras.models.load_model(model_path)
    
    # Load dan preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (224, 224))
    image_normalized = image_resized.astype('float32') / 255.0
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    # Predict
    prediction_prob = model.predict(image_batch, verbose=0)[0][0]
    
    if prediction_prob > 0.5:
        label = "Mask"
        confidence = prediction_prob
    else:
        label = "No Mask"
        confidence = 1 - prediction_prob
    
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")
    
    # Visualize
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(f"{label} ({confidence*100:.2f}%)", fontsize=16)
    plt.axis('off')
    plt.show()
    
    return label, confidence


# ============================================
# EXECUTION
# ============================================

if __name__ == "__main__":
    # PATH DATASET SESUAI STRUKTUR ANDA
    DATASET_PATH = r'C:\Users\User\Mask-Recognition-\dataset'
    
    # Pilih model:
    # False = Transfer Learning (MobileNetV2) - Lebih akurat
    # True = Custom CNN - Lebih cepat
    USE_CUSTOM_CNN = False
    
    # Jalankan training
    model = main_training_pipeline(DATASET_PATH, use_custom_cnn=USE_CUSTOM_CNN)
    
    # Test prediksi (uncomment jika ingin test)
    # test_single_prediction('mask_detector_model.h5', r'C:\Users\User\path\to\test\image.jpg')