"""
Convert MesoInception weights to ONNX by reconstructing the architecture
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
import tf2onnx
import h5py
import numpy as np

# First, let's inspect the H5 file
print("Inspecting MesoInception_DF.h5...")
with h5py.File('MesoInception_DF.h5', 'r') as f:
    def print_structure(name, obj):
        print(name)
    f.visititems(print_structure)

# Define MesoInception architecture based on the original paper
def create_mesoinception():
    """Create MesoInception-4 architecture for deepfake detection"""
    
    inputs = layers.Input(shape=(256, 256, 3), name='input')
    
    # InceptionLayer1
    x1 = layers.Conv2D(1, (1, 1), padding='same', activation='relu')(inputs)
    x2 = layers.Conv2D(4, (1, 1), padding='same', activation='relu')(inputs)
    x2 = layers.Conv2D(4, (3, 3), padding='same', activation='relu')(x2)
    x3 = layers.Conv2D(4, (1, 1), padding='same', activation='relu')(inputs)
    x3 = layers.Conv2D(4, (3, 3), padding='same', activation='relu')(x3)
    x3 = layers.Conv2D(2, (3, 3), padding='same', activation='relu')(x3)
    x = layers.concatenate([x1, x2, x3])
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    
    # InceptionLayer2
    x1 = layers.Conv2D(2, (1, 1), padding='same', activation='relu')(x)
    x2 = layers.Conv2D(4, (1, 1), padding='same', activation='relu')(x)
    x2 = layers.Conv2D(4, (3, 3), padding='same', activation='relu')(x2)
    x3 = layers.Conv2D(4, (1, 1), padding='same', activation='relu')(x)
    x3 = layers.Conv2D(4, (3, 3), padding='same', activation='relu')(x3)
    x3 = layers.Conv2D(2, (3, 3), padding='same', activation='relu')(x3)
    x = layers.concatenate([x1, x2, x3])
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    
    # Conv layers
    x = layers.Conv2D(16, (5, 5), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    
    x = layers.Conv2D(16, (5, 5), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(4, 4), padding='same')(x)
    
    # Dense layers
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(16)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)  # Binary classification
    
    model = Model(inputs=inputs, outputs=outputs, name='MesoInception')
    return model

# Create the model
print("\nCreating MesoInception architecture...")
model = create_mesoinception()

# Try to load weights
try:
    print("Loading weights...")
    model.load_weights('MesoInception_DF.h5')
    print("✅ Weights loaded successfully")
except Exception as e:
    print(f"⚠️ Could not load all weights: {e}")
    print("Proceeding with random weights for demo...")

# Display model summary
model.summary()

# Convert to ONNX
print("\nConverting to ONNX...")
spec = (tf.TensorSpec((None, 256, 256, 3), tf.float32, name="input"),)
output_path = "mesonet4.onnx"

try:
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
    print(f"✅ Model converted successfully to {output_path}")
    print(f"Input shape: [batch_size, 256, 256, 3]")
    print(f"Output shape: [batch_size, 1] (probability of being fake)")
except Exception as e:
    print(f"❌ Conversion error: {e}")
    print("You may need to adjust the model architecture")