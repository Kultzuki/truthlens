"""
Convert MesoInception Keras model to ONNX
"""
import tensorflow as tf
import tf2onnx
import numpy as np

# Load the Keras model
print("Loading MesoInception model...")
model = tf.keras.models.load_model('MesoInception_DF.h5', compile=False)

# Display model summary
model.summary()

# Get input shape
input_shape = model.input_shape
print(f"\nInput shape: {input_shape}")

# Convert to ONNX
print("\nConverting to ONNX...")
spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)
output_path = "mesonet4.onnx"

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)

print(f"✅ Model converted successfully to {output_path}")
print(f"Input name: input")
print(f"Output shape: {model.output_shape}")