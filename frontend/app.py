import streamlit as st
import requests
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="FashionMNIST Classifier",
    page_icon="👕",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("👕 FashionMNIST Image Classifier")
st.caption("Upload a FashionMNIST image and get the predicted class with confidence.")
st.write("Upload a FashionMNIST image and click **Predict**.")

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose a FashionMNIST image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        width=250
    )

    st.write("")

    # -----------------------------
    # Predict Button
    # -----------------------------
    if st.button("🔍 Predict", use_container_width=True):

        with st.spinner("Predicting... Please wait."):

            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    files=files
                )

                if response.status_code == 200:

                    result = response.json()

                    st.success("Prediction Completed Successfully!")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            label="Prediction",
                            value=result["prediction"]
                        )

                    with col2:
                        st.metric(
                            label="Confidence",
                            value=f"{result['confidence']}%"
                        )

                else:
                    st.error("Prediction Failed!")
                    st.write(response.text)

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to FastAPI Backend.")
                st.info("Please start the backend server first.")

            except Exception as e:
                st.error(f"Error: {e}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Developed by Neeraj Nautiyal | FashionMNIST Image Classification")