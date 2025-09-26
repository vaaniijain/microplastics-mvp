import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import tempfile
import math
import matplotlib.pyplot as plt

st.set_page_config(page_title="Microplastics MVP", layout="wide")

st.title("Microplastics Detector — MVP")
st.markdown("""
Upload a microscope image (or smartphone microscopy) of your filter/sample. The app will run simple image processing to **detect candidate microplastic particles**, estimate their sizes (if you provide a scale), and highlight them on the image.

**Quick notes:**
- For best results, use a clear image with high contrast between particles and background.
- If you know the scale, enter microns per pixel (µm/px). If not, sizes are shown in pixels.
- This is an MVP heuristic detector — a proper ML spectral classifier would improve polymer ID.
""")

# Sidebar inputs
st.sidebar.header("Settings")
uploaded = st.sidebar.file_uploader("Upload image (jpg, png, tiff)", type=["jpg","jpeg","png","tif","tiff"])
scale_input = st.sidebar.text_input("Microns per pixel (µm/px) — optional","")
min_area_px = st.sidebar.number_input("Min particle area (px)", min_value=1, value=25)
max_area_px = st.sidebar.number_input("Max particle area (px)", min_value=1, value=1000000)
apply_median = st.sidebar.checkbox("Apply median blur before threshold", value=True)
show_mask = st.sidebar.checkbox("Show binary mask (debug)", value=False)
run_button = st.sidebar.button("Detect")


# Helper functions

def read_image(file) -> np.ndarray:
    image = Image.open(file).convert('RGB')
    return np.array(image)


def preprocess(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    if apply_median:
        gray = cv2.medianBlur(gray, 5)
    # Adaptive threshold to handle uneven illumination
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 9)
    # Morphological opening to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened


def analyze_mask(mask, orig_rgb, min_area, max_area, um_per_px=None):
    # find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    out = orig_rgb.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        # bounding box and moments
        x,y,w,h = cv2.boundingRect(cnt)
        # solidity and circularity heuristics
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 1
        solidity = float(area)/hull_area
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4*math.pi*area/(perimeter*perimeter+1e-6)

        # centroid
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            cx, cy = x + w//2, y + h//2
        else:
            cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])

        # size estimation: equivalent circular diameter
        diameter_px = 2.0 * math.sqrt(area / math.pi)
        diameter_um = diameter_px * um_per_px if um_per_px else None

        # very simple heuristic to flag likely microplastic candidates:
        # - not too elongated (aspect ratio reasonable)
        # - moderate solidity (plastics often have irregular shapes but not extremely fragmented)
        aspect_ratio = float(w)/h if h>0 else 1.0
        score = 0
        # weights for heuristic
        if 0.2 < aspect_ratio < 5.0:
            score += 1
        if solidity > 0.3:
            score += 1
        if circularity > 0.02:
            score += 1
        # area contributes negatively to score if huge
        if area < 200000:
            score += 1

        likely_plastic = score >= 3

        # draw results
        color = (255,0,0) if likely_plastic else (255,165,0)
        cv2.drawContours(out, [cnt], -1, color, 2)
        cv2.rectangle(out, (x,y), (x+w, y+h), color, 1)
        label = f"{int(area)}px"
        if diameter_um:
            label += f", {diameter_um:.1f}µm"
        cv2.putText(out, label, (x, max(y-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        results.append({
            'area_px': area,
            'bbox': (x,y,w,h),
            'centroid': (cx,cy),
            'solidity': solidity,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'diameter_px': diameter_px,
            'diameter_um': diameter_um,
            'score': score,
            'likely_plastic': bool(likely_plastic)
        })

    return out, results


# Main app logic
if uploaded is None:
    st.info("Upload an image to get started.")
else:
    img = read_image(uploaded)
    st.image(img, caption='Original image', use_column_width=True)

    # parse scale
    um_per_px = None
    if scale_input.strip() != "":
        try:
            um_per_px = float(scale_input)
            if um_per_px <= 0:
                st.sidebar.error("Scale must be positive. Ignoring scale.")
                um_per_px = None
        except ValueError:
            st.sidebar.error("Invalid scale. Please enter a number like 0.5 (µm/px).")
            um_per_px = None

    if run_button:
        with st.spinner('Processing...'):
            mask = preprocess(img)
            out_img, items = analyze_mask(mask, img, min_area_px, max_area_px, um_per_px)

        st.success(f"Done — {len(items)} candidate particles detected")

        col1, col2 = st.columns([2,1])
        with col1:
            st.subheader('Detected particles (annotated)')
            st.image(out_img, use_column_width=True)

        with col2:
            st.subheader('Summary')
            detected = [it for it in items if it['likely_plastic']]
            st.markdown(f"**Candidates flagged as likely microplastics:** {len(detected)} / {len(items)}")
            if um_per_px:
                sizes = [it['diameter_um'] for it in detected if it['diameter_um'] is not None]
                if sizes:
                    st.markdown(f"**Size range (µm):** {min(sizes):.1f} — {max(sizes):.1f}")

            st.markdown('---')
            st.markdown('**Detected items (top 20)**')
            for i,it in enumerate(items[:20]):
                tag = 'LIKELY' if it['likely_plastic'] else 'maybe'
                sz = f"{it['diameter_px']:.1f}px"
                if it['diameter_um']:
                    sz += f", {it['diameter_um']:.1f}µm"
                st.text(f"{i+1}. {tag} — area {int(it['area_px'])} px — {sz} — circularity {it['circularity']:.3f}")

        if show_mask:
            st.subheader('Binary mask (debug)')
            st.image(mask, use_column_width=True)

        # histogram of sizes (if we have diameters)
        flagged = [it for it in items if it['likely_plastic'] and it['diameter_um']]
        if flagged:
            fig, ax = plt.subplots()
            dvals = [it['diameter_um'] for it in flagged]
            ax.hist(dvals, bins=12)
            ax.set_xlabel('Diameter (µm)')
            ax.set_ylabel('Count')
            ax.set_title('Size distribution of flagged candidates')
            st.pyplot(fig)

    st.markdown('---')
    st.markdown('**How to improve this MVP:**')
    st.markdown('- Add a trained CNN to classify particles vs debris (use transfer learning on labeled images).')
    st.markdown('- Integrate spectral matching (FTIR / Raman) to confirm polymer type.')
    st.markdown('- Allow batch uploads and a simple database to store results (SQLite).')
    st.markdown('- Provide calibration tools (detect scale bar automatically).')

    st.markdown('---')
    st.caption('This is a heuristic prototype for experimentation and educational use.')
