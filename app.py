# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import cv2 # Required for advanced image processing (cropping/centering)

# --- Function to load and inject custom CSS ---
def inject_css(file_path):
    """Reads a CSS file and injects its content into the Streamlit app."""
    try:
        with open(file_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Could not find {file_path}. Ensure it is in the same directory as app.py.")

# 1. Configuration and Model Loading
st.set_page_config(layout="wide", page_title="Advanced Digit Recognizer")

# Inject the custom CSS for the animated background and UI fixes
inject_css('style.css')

# Use st.cache_resource to load the model only once
@st.cache_resource
def load_trained_model():
    """Load the pre-trained Keras model."""
    MODEL_PATH = 'mnist_cnn_model.h5'
    try:
        with st.spinner("Loading AI model..."):
            model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: Ensure 'mnist_cnn_model.h5' exists (run train_model.py).")
        st.error(f"Underlying error: {e}")
        st.stop()

model = load_trained_model()

st.title("✨ Interactive Handwritten Digit Recognizer")
st.markdown("A highly accurate CNN trained on MNIST. **Draw any digit on the left!**")
st.markdown("---")

# 2. Sidebar for Controls and Information
with st.sidebar:
    st.header("App Controls")
    st.info("💡 Draw robustly! The advanced preprocessing method crops and centers your drawing for best accuracy.")
    
    # Customization Sliders
    stroke_width = st.slider("Pen Thickness", 10, 30, 18)
    canvas_size = st.slider("Canvas Size", 150, 300, 220)
    
    st.markdown("---")
    st.caption("Model Details")
    st.write(f"Architecture: Simple CNN")
    st.write("Input Size: 28x28 (processed)")

# 3. Canvas and Layout (3 Columns)
col_canvas, col_prediction, col_processed = st.columns([1.2, 1, 0.8])

with col_canvas:
    st.subheader("🖍️ Drawing Canvas")
    # Streamlit Drawable Canvas component
    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=canvas_size,
        width=canvas_size,
        drawing_mode="freedraw",
        key="canvas_app",
    )
    st.markdown("<br>", unsafe_allow_html=True) # Spacer

# 4. Prediction Logic (Activated on any draw update)
if canvas_result.image_data is not None:
    img_array_rgba = canvas_result.image_data

    # Check if the canvas is completely blank
    if img_array_rgba.sum() > 0:
        
        # Convert RGBA array to grayscale PIL Image
        pil_img = Image.fromarray(img_array_rgba.astype('uint8'), 'RGBA').convert('L')
        img_np = np.array(pil_img)
        
        # --- Advanced Preprocessing (Robust Cropping and Centering) ---
        
        # Find the bounding box of the drawn digit
        # Uses a threshold of 50 to identify drawn pixels
        _, thresh = cv2.threshold(img_np, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Crop the image to the bounding box
            cropped_img_np = img_np[y:y+h, x:x+w]
            
            # Rescale the cropped area to fit within a 20x20 area, maintaining aspect ratio
            target_size = 20
            scale = target_size / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            resized_img = cv2.resize(cropped_img_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Create a 28x28 empty canvas (MNIST format)
            mnist_canvas = np.zeros((28, 28), dtype=np.float32)

            # Center the resized image on the 28x28 canvas
            start_x = (28 - new_w) // 2
            start_y = (28 - new_h) // 2
            
            mnist_canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized_img / 255.0
            
            # Prepare tensor for prediction
            input_tensor = mnist_canvas.reshape(1, 28, 28, 1)

            # --- Prediction ---
            predictions = model.predict(input_tensor, verbose=0)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions) * 100

            # --- Display Results ---
            with col_prediction:
                st.subheader("💡 Model Prediction")
                st.success(f"**Predicted Digit:**")
                st.metric(label="Result", value=predicted_class, delta=f"{confidence:.2f}% Confidence")
                
                # Probability Distribution Chart
                st.markdown("#### Probability Distribution")
                labels = [str(i) for i in range(10)]
                chart_data = {"Digit": labels, "Probability": predictions[0]}
                st.bar_chart(chart_data, x='Digit', y='Probability', color="#ffaa00")

            # --- Display Processed Image ---
            with col_processed:
                st.subheader("⚙️ Processed Input")
                # Convert 28x28 input back to image for display
                display_img = Image.fromarray((mnist_canvas * 255).astype(np.uint8), mode='L')
                st.image(display_img, caption="Input (28x28) fed to CNN.", width=100)
                
        else:
            with col_prediction:
                st.info("Please draw a visible digit to get a prediction.")
    else:
        with col_prediction:
            st.info("Start drawing a digit on the left canvas.")
