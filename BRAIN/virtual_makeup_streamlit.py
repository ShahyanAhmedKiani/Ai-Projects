"""
Virtual Makeup Studio - Streamlit App (Python)
Features:
- Upload an image and apply virtual makeup (lipstick, blush, simple eyeshadow)
- Real-time webcam mode with overlays
- Hand-control gestures (pinch to toggle lipstick, pinch two fingers to cycle colors)
  using MediaPipe Hands when webcam mode is active.
Notes/requirements:
- Install dependencies: streamlit, opencv-python, mediapipe, numpy, pillow
- Run: streamlit run virtual_makeup_streamlit.py
- This is a starting point; performance will vary by machine.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import mediapipe as mp
import time

st.set_page_config(page_title='Virtual Makeup Studio', layout='wide')

# ---- Helpers ----
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

def np_img_from_pil(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def pil_from_np_img(np_img):
    return Image.fromarray(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))

# Lip and cheek landmark indices (MediaPipe Face Mesh)
# These lists are approximate and taken from the Face Mesh 468 landmarks mapping.
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
LEFT_CHEEK = 234  # general area around left cheek (approx)
RIGHT_CHEEK = 454  # right cheek approximate

# Predefined lipstick colors
DEFAULT_LIP_COLORS = [
    "#d43f5a", "#b22c6f", "#ff6b6b", "#ff9f1c", "#c56cf0", "#7f3fbf"
]

def hex_to_bgr(hex_color):
    h = hex_color.lstrip('#')
    r, g, b = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)

def blend_overlay(base_bgr, overlay_bgr, mask_alpha):
    """Blend overlay color into base image (BGR numpy arrays). mask_alpha in [0..1]."""
    return (base_bgr * (1 - mask_alpha) + overlay_bgr * mask_alpha).astype(np.uint8)

def draw_filled_poly(img, points, color_bgr, alpha=0.6, feather=8):
    """Draw a filled polygon on img with soft edges by using a mask and Gaussian blur."""
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    pts = np.array(points, np.int32)
    cv2.fillPoly(mask, [pts], 255)
    # feather edges
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (feather*2+1, feather*2+1), 0)
    colored = np.zeros_like(img, dtype=np.uint8)
    colored[:] = color_bgr
    alpha_mask = (mask.astype(np.float32) / 255.0) * alpha
    # blend per channel
    out = img.copy().astype(np.float32)
    for c in range(3):
        out[:, :, c] = out[:, :, c] * (1 - alpha_mask) + colored[:, :, c] * alpha_mask
    return out.astype(np.uint8)

def landmarks_to_points(landmarks, image_w, image_h, idx_list):
    pts = []
    for idx in idx_list:
        lm = landmarks[idx]
        x, y = int(lm.x * image_w), int(lm.y * image_h)
        pts.append((x, y))
    return pts

# ---- UI ----
st.title("🎨 Virtual Makeup Studio (Python / Streamlit)")
col1, col2 = st.columns([2, 1])

with col2:
    st.header("Controls")
    mode = st.radio("Mode", ("Image Upload", "Webcam (Real-time)"))
    st.markdown("**Makeup options**")
    lip_color = st.color_picker("Lip Color", "#d43f5a")
    blush_color = st.color_picker("Blush Color", "#ff9aa2")
    intensity = st.slider("Intensity", min_value=0.0, max_value=1.0, value=0.6, step=0.01)
    apply_lip = st.checkbox("Apply Lipstick", value=True)
    apply_blush = st.checkbox("Apply Blush", value=True)
    apply_eyeshadow = st.checkbox("Apply Eyeshadow (simple)", value=True)
    hand_control_enabled = st.checkbox("Enable Hand Control (Webcam only)", value=False)
    st.markdown("---")
    st.write("Hand gestures (webcam):\n- Pinch (thumb+index) -> toggle lipstick\n- Pinch (index+middle) -> cycle lip color")

with col1:
    if mode == "Image Upload":
        uploaded = st.file_uploader("Upload a face image", type=['png','jpg','jpeg'])
        if uploaded is not None:
            img_pil = Image.open(uploaded).convert("RGB")
            img_bgr = np_img_from_pil(img_pil)
            st.image(img_pil, caption="Original image", use_column_width=True)
            # Process static image
            with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                       refine_landmarks=True,
                                       min_detection_confidence=0.5) as face_mesh:
                rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                if not results.multi_face_landmarks:
                    st.warning("No face detected. Try a clearer frontal face image.")
                else:
                    face_landmarks = results.multi_face_landmarks[0].landmark
                    h, w = img_bgr.shape[:2]
                    out = img_bgr.copy()
                    # lipstick
                    if apply_lip:
                        lip_pts = landmarks_to_points(face_landmarks, w, h, LIPS_OUTER)
                        lip_color_bgr = hex_to_bgr(lip_color)
                        out = draw_filled_poly(out, lip_pts, lip_color_bgr, alpha=intensity, feather=10)
                    # blush
                    if apply_blush:
                        # approximate cheek centers
                        lc = face_landmarks[LEFT_CHEEK]
                        rc = face_landmarks[RIGHT_CHEEK]
                        lx, ly = int(lc.x * w), int(lc.y * h)
                        rx, ry = int(rc.x * w), int(rc.y * h)
                        # draw soft ellipses
                        overlay = out.copy()
                        bgr = hex_to_bgr(blush_color)
                        # left
                        mask = np.zeros((h,w), dtype=np.uint8)
                        cv2.ellipse(mask, (lx, ly), (int(w*0.06), int(h*0.04)), 0, 0, 360, 255, -1)
                        mask = cv2.GaussianBlur(mask, (0,0), sigmaX=15, sigmaY=15)
                        alpha_mask = (mask.astype(np.float32) / 255.0) * (intensity*0.9)
                        for c in range(3):
                            overlay[:, :, c] = overlay[:, :, c] * (1 - alpha_mask) + bgr[c] * alpha_mask
                        out = overlay
                        # right
                        overlay = out.copy()
                        mask = np.zeros((h,w), dtype=np.uint8)
                        cv2.ellipse(mask, (rx, ry), (int(w*0.06), int(h*0.04)), 0, 0, 360, 255, -1)
                        mask = cv2.GaussianBlur(mask, (0,0), sigmaX=15, sigmaY=15)
                        alpha_mask = (mask.astype(np.float32) / 255.0) * (intensity*0.9)
                        for c in range(3):
                            overlay[:, :, c] = overlay[:, :, c] * (1 - alpha_mask) + bgr[c] * alpha_mask
                        out = overlay
                    # simple eyeshadow
                    if apply_eyeshadow:
                        # use left/right eye landmarks (a few points)
                        left_eye_idx = [33, 246, 161, 160, 159, 158, 157, 173]
                        right_eye_idx = [362, 398, 384, 385, 386, 387, 388, 466]
                        le_pts = landmarks_to_points(face_landmarks, w, h, left_eye_idx)
                        re_pts = landmarks_to_points(face_landmarks, w, h, right_eye_idx)
                        shadow_color = hex_to_bgr("#a37bff")
                        out = draw_filled_poly(out, le_pts, shadow_color, alpha=intensity*0.25, feather=8)
                        out = draw_filled_poly(out, re_pts, shadow_color, alpha=intensity*0.25, feather=8)
                    st.image(pil_from_np_img(out), caption="With Makeup", use_column_width=True)
    else:
        run = st.button("Start Webcam")
        stop = st.button("Stop Webcam")
        video_placeholder = st.empty()
        if run:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot open webcam")
            else:
                # Hand detection and face mesh in realtime
                face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                 refine_landmarks=True,
                                                 min_detection_confidence=0.5,
                                                 min_tracking_confidence=0.5)
                hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                                       min_detection_confidence=0.5,
                                       min_tracking_confidence=0.5)
                current_lip_colors = DEFAULT_LIP_COLORS.copy()
                current_color_idx = 0
                lip_on = apply_lip
                prev_pinch_time = 0
                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = cv2.flip(frame, 1)  # mirror
                        h, w = frame.shape[:2]
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        face_results = face_mesh.process(rgb)
                        # hands processing for gestures
                        if hand_control_enabled:
                            hand_results = hands.process(rgb)
                            if hand_results.multi_hand_landmarks:
                                hand_landmarks = hand_results.multi_hand_landmarks[0]
                                # thumb tip (4) and index tip (8)
                                thumb = hand_landmarks.landmark[4]
                                index = hand_landmarks.landmark[8]
                                middle = hand_landmarks.landmark[12]
                                # normalized distance
                                dx = thumb.x - index.x
                                dy = thumb.y - index.y
                                dist = np.hypot(dx, dy)
                                dx2 = index.x - middle.x
                                dy2 = index.y - middle.y
                                dist2 = np.hypot(dx2, dy2)
                                now = time.time()
                                # pinch thumb+index -> toggle lipstick (debounced)
                                if dist < 0.05 and now - prev_pinch_time > 0.6:
                                    lip_on = not lip_on
                                    prev_pinch_time = now
                                # pinch index+middle -> cycle color
                                if dist2 < 0.04 and now - prev_pinch_time > 0.6:
                                    current_color_idx = (current_color_idx + 1) % len(current_lip_colors)
                                    prev_pinch_time = now
                                # draw simple hand landmarks for debugging
                                for lm in hand_landmarks.landmark:
                                    cx, cy = int(lm.x * w), int(lm.y * h)
                                    cv2.circle(frame, (cx, cy), 2, (0,255,0), -1)
                        else:
                            hand_results = None
                        out = frame.copy()
                        if face_results.multi_face_landmarks:
                            face_landmarks = face_results.multi_face_landmarks[0].landmark
                            # lips
                            if apply_lip and lip_on:
                                lip_pts = landmarks_to_points(face_landmarks, w, h, LIPS_OUTER)
                                lip_color_to_use = current_lip_colors[current_color_idx]
                                out = draw_filled_poly(out, lip_pts, hex_to_bgr(lip_color_to_use), alpha=intensity, feather=8)
                            # blush
                            if apply_blush:
                                lc = face_landmarks[LEFT_CHEEK]
                                rc = face_landmarks[RIGHT_CHEEK]
                                lx, ly = int(lc.x * w), int(lc.y * h)
                                rx, ry = int(rc.x * w), int(rc.y * h)
                                overlay = out.copy()
                                bgr = hex_to_bgr(blush_color)
                                mask = np.zeros((h,w), dtype=np.uint8)
                                cv2.ellipse(mask, (lx, ly), (int(w*0.06), int(h*0.04)), 0, 0, 360, 255, -1)
                                mask = cv2.GaussianBlur(mask, (0,0), sigmaX=15, sigmaY=15)
                                alpha_mask = (mask.astype(np.float32) / 255.0) * (intensity*0.9)
                                for c in range(3):
                                    overlay[:, :, c] = overlay[:, :, c] * (1 - alpha_mask) + bgr[c] * alpha_mask
                                out = overlay
                                overlay = out.copy()
                                mask = np.zeros((h,w), dtype=np.uint8)
                                cv2.ellipse(mask, (rx, ry), (int(w*0.06), int(h*0.04)), 0, 0, 360, 255, -1)
                                mask = cv2.GaussianBlur(mask, (0,0), sigmaX=15, sigmaY=15)
                                alpha_mask = (mask.astype(np.float32) / 255.0) * (intensity*0.9)
                                for c in range(3):
                                    overlay[:, :, c] = overlay[:, :, c] * (1 - alpha_mask) + bgr[c] * alpha_mask
                                out = overlay
                            # eyeshadow
                            if apply_eyeshadow:
                                left_eye_idx = [33, 246, 161, 160, 159, 158, 157, 173]
                                right_eye_idx = [362, 398, 384, 385, 386, 387, 388, 466]
                                le_pts = landmarks_to_points(face_landmarks, w, h, left_eye_idx)
                                re_pts = landmarks_to_points(face_landmarks, w, h, right_eye_idx)
                                out = draw_filled_poly(out, le_pts, hex_to_bgr("#a37bff"), alpha=intensity*0.25, feather=6)
                                out = draw_filled_poly(out, re_pts, hex_to_bgr("#a37bff"), alpha=intensity*0.25, feather=6)
                        # display
                        video_placeholder.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), channels='RGB')
                        # stop condition check
                        if stop:
                            break
                        # allow Streamlit to be responsive
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                except Exception as e:
                    st.error(f\"Webcam loop error: {e}\")
                finally:
                    cap.release()
                    face_mesh.close()
                    hands.close()
                    st.info(\"Webcam stopped.\")
        else:
            st.info(\"Switch to Webcam mode and press Start Webcam to begin.\")


st.markdown(\"---\")
st.markdown(\"**Tips & Notes**:\") 
st.markdown(\"- This demo uses MediaPipe Face Mesh to get facial landmarks and places simple makeup overlays.\\n- Hand control uses MediaPipe Hands and simple distance thresholds; tune thresholds for your camera and environment.\\n- If the webcam loop seems slow, reduce frame size or disable hand control.\") 
