import os
import json
import base64
import numpy as np
import cv2
import time
import traceback
from flask import Flask, request, jsonify

# Firebase Gen 2 SDK
from firebase_functions import https_fn, options
from firebase_admin import initialize_app

# OCR Dependencies
from flask_cors import CORS
from rapidocr_onnxruntime import RapidOCR

# Initialize Firebase Admin
initialize_app()

APP_NAME = "ocr"

# --- Configuration & Initialization ---

CONFIG_FILE = 'config.json'
DEFAULT_CONFIG = {
    "preprocessing": {
        "apply_contrast": False,
        "contrast_alpha": 1.0,
        "contrast_beta": 0,
        "apply_gray": False,
        "apply_threshold": False,
        "threshold_block_size": 15,
        "threshold_c": 5
    }
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return DEFAULT_CONFIG

config = load_config()

# LAZY INITIALIZATION: Don't initialize OCR engine at module load time
# This prevents timeout during Firebase CLI deployment discovery
rapid_ocr_engine = None

def get_rapid_ocr():
    global rapid_ocr_engine
    if rapid_ocr_engine is None:
        print("Initializing RapidOCR on first request...")
        rapid_ocr_engine = RapidOCR()
    return rapid_ocr_engine


# --- Helper Functions ---

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def detect_and_crop_to_content(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        
        if not cnts:
            return False, image

        min_area = 1000 
        large_cnts = [c for c in cnts if cv2.contourArea(c) > min_area]

        if not large_cnts:
             print("No large contours found. Returning original.")
             return False, image

        x_min, y_min = np.inf, np.inf
        x_max, y_max = -np.inf, -np.inf

        for c in large_cnts:
            x, y, w, h = cv2.boundingRect(c)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        padding = 10
        h_img, w_img = image.shape[:2]
        
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w_img, x_max + padding)
        y_max = min(h_img, y_max + padding)

        cropped = image[int(y_min):int(y_max), int(x_min):int(x_max)]
        print(f"Content detected. Cropped to: {x_min},{y_min} - {x_max},{y_max}")
        return True, cropped

    except Exception as e:
        print(f"Error in content cropping: {e}")
        return False, image

def apply_sharpening(image):
    kernel = np.array([[0, -1, 0], 
                       [-1, 5,-1], 
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def sort_boxes(dt_boxes, matched_texts, matched_scores):
    if dt_boxes is None or len(dt_boxes) == 0:
        return [], [], []

    items = []
    for i in range(len(dt_boxes)):
        box = dt_boxes[i]
        box = np.array(box, dtype=np.float32)
        if box.ndim == 1:
            if box.size == 8:
                 box = box.reshape((4, 2))
            elif box.size == 4:
                x_min, y_min, x_max, y_max = box
                box = np.array([
                    [x_min, y_min],
                    [x_max, y_min],
                    [x_max, y_max],
                    [x_min, y_max]
                ], dtype=np.float32)
            else:
                 print(f"Skipping box with unexpected shape: {box.shape}")
                 continue
        
        items.append({
            'box': box,
            'text': matched_texts[i],
            'score': matched_scores[i] if matched_scores else 1.0,
            'y_top': min(box[0][1], box[1][1]), # Top Y
            'x_left': min(box[0][0], box[3][0]) # Left X
        })

    items.sort(key=lambda x: x['y_top'])

    lines = []
    current_line = []
    
    if items:
        current_line.append(items[0])
        for i in range(1, len(items)):
            item = items[i]
            prev_item = current_line[-1]
            prev_height = abs(prev_item['box'][2][1] - prev_item['box'][0][1])
            threshold = prev_height * 0.5 
            
            if abs(item['y_top'] - prev_item['y_top']) < threshold:
                current_line.append(item)
            else:
                lines.append(current_line)
                current_line = [item]
        lines.append(current_line)

    sorted_items = []
    for line in lines:
        line.sort(key=lambda x: x['x_left'])
        sorted_items.extend(line)

    sorted_boxes = [x['box'] for x in sorted_items]
    sorted_texts = [x['text'] for x in sorted_items]
    sorted_scores = [x['score'] for x in sorted_items]

    return sorted_boxes, sorted_texts, sorted_scores

# --- Flask App Definition ---

app = Flask(__name__)
CORS(app)

@app.route('/ocr', methods=['POST'])
def process_image():
    # ... (Same logic as before, just wrapped in Flask route)
    default_prep_config = config.get('preprocessing', {})

    try:
        data = request.get_json(silent=True)
        if not data:
             data = request.form

        if not data or ('image' not in data and 'file' not in request.files):
            return jsonify({'error': 'No image data provided'}), 400

        # Handle image input
        image_data = None
        if 'image' in data:
            image_data = data['image']
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif 'file' in request.files:
            file = request.files['file']
            nparr = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
             return jsonify({'error': 'Failed to decode image'}), 400

        # Initialize config
        request_prep = data.get('preprocessing', {})
        if isinstance(request_prep, str):
            try:
                request_prep = json.loads(request_prep)
            except:
                request_prep = {}
        
        prep_config = {**default_prep_config, **request_prep}

        print(f"Applying preprocessing: {prep_config}")
        img_to_process = img.copy()

        # 1. Deskew / Crop
        if prep_config.get('apply_deskew', False):
             success, cropped_img = detect_and_crop_to_content(img_to_process)
             if success:
                 img_to_process = cropped_img

        # 1.5 Resize
        if prep_config.get('apply_resize', False):
            target_h = int(prep_config.get('resize_height', 2000))
            h, w = img_to_process.shape[:2]
            if h > target_h:
                scale = target_h / h
                new_w = int(w * scale)
                img_to_process = cv2.resize(img_to_process, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
                print(f"Downresized to {new_w}x{target_h}")

        # 2. Contrast
        if prep_config.get('apply_contrast', False):
            alpha = float(prep_config.get('contrast_alpha', 1.5))
            beta = float(prep_config.get('contrast_beta', 0))
            img_to_process = cv2.convertScaleAbs(img_to_process, alpha=alpha, beta=beta)

        # 3. Sharpening
        if prep_config.get('apply_sharpening', False):
            img_to_process = apply_sharpening(img_to_process)

        # 4. Grayscale
        if prep_config.get('apply_gray', False):
             if len(img_to_process.shape) == 3:
                img_to_process = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2GRAY)
                img_to_process = cv2.cvtColor(img_to_process, cv2.COLOR_GRAY2BGR)

        # 5. Thresholding
        if prep_config.get('apply_threshold', False):
            gray_for_thresh = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2GRAY)
            block_size = int(prep_config.get('threshold_block_size', 15))
            c_val = int(prep_config.get('threshold_c', 5))
            if block_size % 2 == 0: block_size += 1
            img_to_process = cv2.adaptiveThreshold(
                gray_for_thresh, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                block_size, 
                c_val
            )
            img_to_process = cv2.cvtColor(img_to_process, cv2.COLOR_GRAY2BGR)

        # --- OCR ---
        detected_text = ""
        debug_img = img_to_process.copy()
        
        start_time = time.time()

        # Use RapidOCR
        ocr_engine = get_rapid_ocr()
        
        result, elapse = ocr_engine(img_to_process)
        if result:
            boxes = [line[0] for line in result]
            texts = [line[1] for line in result]
            scores = [line[2] for line in result]
            
            sorted_boxes, sorted_texts, _ = sort_boxes(boxes, texts, scores)

            for i in range(len(sorted_texts)):
                text = sorted_texts[i]
                box = sorted_boxes[i]
                detected_text += text + "\n"
                points = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(debug_img, [points], isClosed=True, color=(0, 0, 255), thickness=2)
        else:
            print("No text detected by RapidOCR.")

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"OCR Execution Time: {execution_time:.4f} seconds")

        _, buffer = cv2.imencode('.jpg', debug_img)
        processed_image_b64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'text': detected_text,
            'processed_image': f"data:image/jpeg;base64,{processed_image_b64}",
            'execution_time': execution_time
        })

    except Exception as e:
        print(f"Error processing image: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Expose Flask App via Cloud Function
@https_fn.on_request(
    memory=options.MemoryOption.GB_2,
    timeout_sec=300,
    cors=options.CorsOptions(
        cors_origins="*",
        cors_methods=["GET", "POST", "OPTIONS"],
    ),
)
def ocr(req: https_fn.Request) -> https_fn.Response:
    with app.request_context(req.environ):
        return app.full_dispatch_request()
