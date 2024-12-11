import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ruta del dataset
data_dir = "C:/Users/VRNK1/Documents/python/dataset"

# Parámetros del modelo
img_size = (64, 64)
batch_size = 32

# Preparación del dataset con ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

# Generador de entrenamiento
train_generator = train_datagen.flow_from_directory(
    os.path.join(data_dir, 'train'),  # Ruta a la carpeta de entrenamiento
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Generador de validación
validation_generator = train_datagen.flow_from_directory(
    os.path.join(data_dir, 'validation'),  # Ruta a la carpeta de validación
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Crear el modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(train_generator.class_indices), activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

# Guardar el modelo
model.save('vehicle_classifier_model.h5')

# Función para predecir imágenes
def predict_vehicle(image_path, model):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("La imagen no pudo ser cargada. Verifica la ruta.")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convierte de BGR a RGB
    image = cv2.resize(image, img_size)  # Cambia el tamaño de la imagen
    image = np.expand_dims(image, axis=0)  # Añade la dimensión del batch
    image = image / 255.0  # Normaliza la imagen

    # Realiza la predicción
    predictions = model.predict(image)
    class_idx = np.argmax(predictions)  # Obtén el índice de la clase con mayor probabilidad
    return class_idx, predictions[0]

# Cargar el modelo
model = tf.keras.models.load_model('vehicle_classifier_model.h5')

# Mapeo de clases
class_labels = list(train_generator.class_indices.keys())  # Asegúrate que las clases estén bien mapeadas

# Predicción desde una imagen
image_path = "C:/Users/VRNK1/Documents/python/dataset/test_image.jpg"  # Cambia por la ruta de tu imagen
class_idx, probabilities = predict_vehicle(image_path, model)
print(f"La imagen pertenece a la clase: {class_labels[class_idx]} con probabilidad {probabilities[class_idx]:.2f}")

# Predicción desde cámara web
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara")
        break

    # Procesar la imagen capturada
    resized_frame = cv2.resize(frame, img_size)
    normalized_frame = resized_frame / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=0)

    # Predicción
    predictions = model.predict(input_frame)
    class_idx = np.argmax(predictions)
    label = class_labels[class_idx]

    # Mostrar el resultado en la imagen
    cv2.putText(frame, f"{label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Vehicle Classifier', frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
