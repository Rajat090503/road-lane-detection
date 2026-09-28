import streamlit as st
import cv2
import numpy as np
import tempfile
import os

st.set_page_config(
    page_title="Road Lane Detection",
    page_icon="🛣️",
    layout="wide"
)

st.title("🛣️ Road Lane Detection")
st.write("Detect road lane lines using Python, OpenCV and Computer Vision.")

# -----------------------------
# Lane Detection Functions
# -----------------------------

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, 50, 150)


def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]

    triangle = np.array([[
        (int(width * 0.10), height),
        (int(width * 0.90), height),
        (int(width * 0.50), int(height * 0.55))
    ]])

    mask = np.zeros_like(image)

    if len(image.shape) == 2:
        cv2.fillPoly(mask, triangle, 255)
    else:
        cv2.fillPoly(mask, triangle, (255, 255, 255))

    return cv2.bitwise_and(image, mask)


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters

    height = image.shape[0]
    y1 = height
    y2 = int(height * 0.60)

    if abs(slope) < 0.001:
        return None

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []

    if lines is None:
        return []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)

        if x1 == x2:
            continue

        parameters = np.polyfit(
            (x1, x2),
            (y1, y2),
            1
        )

        slope = parameters[0]
        intercept = parameters[1]

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    lane_lines = []

    if left_fit:
        left_average = np.average(left_fit, axis=0)
        left_line = make_coordinates(image, left_average)

        if left_line is not None:
            lane_lines.append(left_line)

    if right_fit:
        right_average = np.average(right_fit, axis=0)
        right_line = make_coordinates(image, right_average)

        if right_line is not None:
            lane_lines.append(right_line)

    return lane_lines


def display_lines(image, lines):
    line_image = np.zeros_like(image)

    if lines is not None:
        for line in lines:
            if line is not None:
                x1, y1, x2, y2 = line.reshape(4)

                cv2.line(
                    line_image,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    10
                )

    return line_image


def detect_lanes(frame):
    edges = canny(frame)

    cropped = region_of_interest(edges)

    lines = cv2.HoughLinesP(
        cropped,
        2,
        np.pi / 180,
        100,
        np.array([]),
        minLineLength=40,
        maxLineGap=5
    )

    averaged_lines = average_slope_intercept(
        frame,
        lines
    )

    line_image = display_lines(
        frame,
        averaged_lines
    )

    result = cv2.addWeighted(
        frame,
        0.8,
        line_image,
        1,
        1
    )

    return result


# -----------------------------
# User Interface
# -----------------------------

st.sidebar.header("Upload Road Image")

uploaded_file = st.sidebar.file_uploader(
    "Choose a road image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if image is None:
        st.error("Unable to read the uploaded image.")
    else:

        result = detect_lanes(image)

        original_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        result_rgb = cv2.cvtColor(
            result,
            cv2.COLOR_BGR2RGB
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(
                original_rgb,
                use_container_width=True
            )

        with col2:
            st.subheader("Detected Lane Lines")
            st.image(
                result_rgb,
                use_container_width=True
            )

        # Download result
        success, encoded_image = cv2.imencode(
            ".jpg",
            result
        )

        if success:
            st.download_button(
                label="⬇️ Download Result",
                data=encoded_image.tobytes(),
                file_name="lane_detection_result.jpg",
                mime="image/jpeg"
            )

else:

    st.info(
        "👈 Upload a road image from the sidebar to detect lane lines."
    )

    st.subheader("📌 How it works")

    st.markdown("""
    1. Convert image to grayscale
    2. Apply Gaussian Blur
    3. Detect edges using Canny
    4. Apply Region of Interest
    5. Detect lines using Hough Transform
    6. Calculate average lane lines
    7. Display detected lanes
    """)

    st.subheader("🛠️ Technologies")

    st.write(
        "Python • OpenCV • NumPy • Streamlit"
    )
