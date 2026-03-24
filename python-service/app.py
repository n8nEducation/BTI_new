from flask import Flask, request, send_file
from pdf2image import convert_from_bytes
from PIL import Image, ImageDraw
import io
import json
import math
import cv2
import numpy as np

app = Flask(__name__)


def region_to_polygon(region, width, height):
    """Convert region data to list of pixel (x, y) tuples.
    Supports:
      - polygon: [[x,y], ...] in 0.0–1.0          (new format, 4-6 points)
      - {x1,y1,x2,y2} in 0.0–1.0                  (legacy bbox)
      - {x,y,w,h} in 0–100                          (legacy bbox %)
    """
    if isinstance(region, list):
        return [(int(p[0] * width), int(p[1] * height)) for p in region]
    if isinstance(region, dict):
        if 'x1' in region:
            x1 = int(region['x1'] * width)
            y1 = int(region['y1'] * height)
            x2 = int(region['x2'] * width)
            y2 = int(region['y2'] * height)
            return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        x = int(region.get('x', 0) / 100 * width)
        y = int(region.get('y', 0) / 100 * height)
        w = int(region.get('w', 10) / 100 * width)
        h = int(region.get('h', 10) / 100 * height)
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    return []


def region_to_polygon_raw(region):
    """Return polygon points as normalized [[x,y], ...] in 0.0-1.0, or None."""
    if isinstance(region, list) and region:
        return region
    if isinstance(region, dict) and 'x1' in region:
        return [[region['x1'], region['y1']], [region['x2'], region['y1']],
                [region['x2'], region['y2']], [region['x1'], region['y2']]]
    return None


def polygon_centroid(pts):
    """Return (cx, cy) centroid of a polygon."""
    cx = sum(p[0] for p in pts) // len(pts)
    cy = sum(p[1] for p in pts) // len(pts)
    return cx, cy


def find_shared_edge(poly1, poly2, width, height, tol=0.05):
    """Find the shared boundary segment between two room polygons.
    Expects normalized 0.0-1.0 polygon points.
    Returns pixel (x1,y1,x2,y2) of the longest matching edge, or None."""
    def pt_dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    best, best_len = None, 0
    for i in range(len(poly1)):
        a1 = poly1[i]
        b1 = poly1[(i + 1) % len(poly1)]
        for j in range(len(poly2)):
            a2 = poly2[j]
            b2 = poly2[(j + 1) % len(poly2)]
            match = (pt_dist(a1, a2) < tol and pt_dist(b1, b2) < tol) or \
                    (pt_dist(a1, b2) < tol and pt_dist(b1, a2) < tol)
            if match:
                seg_len = pt_dist(a1, b1)
                if seg_len > best_len:
                    best_len = seg_len
                    best = (int(a1[0] * width), int(a1[1] * height),
                            int(b1[0] * width), int(b1[1] * height))
    return best


def find_closest_edge_pair(poly1, poly2):
    """Find the closest pair of edges (by midpoint distance) between two normalized polygons.
    Returns (a1, b1, a2, b2) or None."""
    best = None
    best_d = float('inf')
    for i in range(len(poly1)):
        a1, b1 = poly1[i], poly1[(i + 1) % len(poly1)]
        m1 = ((a1[0] + b1[0]) / 2, (a1[1] + b1[1]) / 2)
        for j in range(len(poly2)):
            a2, b2 = poly2[j], poly2[(j + 1) % len(poly2)]
            m2 = ((a2[0] + b2[0]) / 2, (a2[1] + b2[1]) / 2)
            d = math.hypot(m1[0] - m2[0], m1[1] - m2[1])
            if d < best_d:
                best_d = d
                best = (a1, b1, a2, b2)
    return best


def find_nearest_hough_line(img_cv, qx, qy, radius=80):
    """Find the Hough line segment closest to pixel (qx, qy) within radius px.
    Returns (x1, y1, x2, y2) or None."""
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, threshold=40,
                             minLineLength=20, maxLineGap=15)
    if lines is None:
        return None

    def dist_pt_seg(px, py, x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return math.hypot(px - x1, py - y1)
        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        return math.hypot(px - x1 - t * dx, py - y1 - t * dy)

    best_line = None
    best_d = float('inf')
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        d = dist_pt_seg(qx, qy, x1, y1, x2, y2)
        if d < best_d:
            best_d = d
            best_line = (x1, y1, x2, y2)

    return best_line if best_d <= radius else None


@app.route('/health', methods=['GET'])
def health():
    return {'status': 'ok'}


@app.route('/convert-pdf', methods=['POST'])
def convert_pdf():
    if 'file' not in request.files:
        return {'error': 'No file provided'}, 400
    pdf_bytes = request.files['file'].read()
    if not pdf_bytes:
        return {'error': 'Empty file'}, 400
    try:
        images = convert_from_bytes(pdf_bytes, first_page=1, last_page=1, dpi=200, use_pdftocairo=True)
    except Exception:
        images = convert_from_bytes(pdf_bytes, first_page=1, last_page=1, dpi=200)
    if not images:
        return {'error': 'Failed to convert PDF'}, 500
    img_io = io.BytesIO()
    images[0].save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png', download_name='plan.png')


@app.route('/crop-plan', methods=['POST'])
def crop_plan():
    """
    Finds the largest closed contour (apartment outline) and returns a cropped image.
    After cropping, plan occupies 100% of the frame — GPT coordinates 0.0-1.0 map to real walls.
    Input:  multipart/form-data { image: <PNG binary> }
    Output: PNG binary (cropped)
    """
    if 'image' not in request.files:
        return {'error': 'No image provided'}, 400

    img_bytes = request.files['image'].read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_cv is None:
        return {'error': 'Failed to decode image'}, 400

    h, w = img_cv.shape[:2]
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Threshold: floor plans are usually dark lines on white background
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Dilate to connect broken lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=3)

    # Find external contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Fallback: return original
        img_io = io.BytesIO(img_bytes)
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png', download_name='cropped_plan.png')

    # Pick the largest contour by area, ignoring tiny noise
    min_area = (w * h) * 0.05  # at least 5% of image
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid:
        img_io = io.BytesIO(img_bytes)
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png', download_name='cropped_plan.png')

    largest = max(valid, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)

    # Add margin (1% of each dimension)
    margin_x = max(10, int(w * 0.01))
    margin_y = max(10, int(h * 0.01))
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(w, x + bw + margin_x)
    y2 = min(h, y + bh + margin_y)

    # Sanity check: cropped area must be at least 30% of original
    if (x2 - x1) * (y2 - y1) < w * h * 0.30:
        img_io = io.BytesIO(img_bytes)
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png', download_name='cropped_plan.png')

    cropped_cv = img_cv[y1:y2, x1:x2]
    _, buf = cv2.imencode('.png', cropped_cv)
    img_io = io.BytesIO(buf.tobytes())
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png', download_name='cropped_plan.png')


@app.route('/annotate-rooms', methods=['POST'])
def annotate_rooms():
    """
    Draws semi-transparent room labels on the floor plan.
    Input:  multipart/form-data  { image: <PNG binary>, rooms_json: <JSON string> }
    Output: PNG binary
    """
    if 'image' not in request.files:
        return {'error': 'No image provided'}, 400
    if 'rooms_json' not in request.form:
        return {'error': 'No rooms_json provided'}, 400

    img = Image.open(request.files['image']).convert('RGBA')
    rooms = json.loads(request.form['rooms_json'])
    width, height = img.size

    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for i, room in enumerate(rooms):
        poly = region_to_polygon(room.get('polygon') or room.get('region_percent', {}), width, height)
        if not poly:
            continue

        cx, cy = polygon_centroid(poly)
        r = max(12, int(min(width, height) * 0.018))

        # Numbered circle badge at room centroid
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(59, 130, 246, 220))
        num = str(i + 1)
        draw.text((cx - r // 2, cy - r // 2), num, fill=(255, 255, 255, 255))

        # Room name to the right of the badge
        label = room.get('name', f'Помещение {i + 1}')
        draw.text((cx + r + 4, cy - r // 2), label, fill=(0, 0, 100, 230))

    result = Image.alpha_composite(img, overlay).convert('RGB')
    img_io = io.BytesIO()
    result.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png', download_name='annotated_rooms.png')


@app.route('/annotate-changes', methods=['POST'])
def annotate_changes():
    """
    Draws wall-level annotations on the floor plan.
    Input:  multipart/form-data  { image: <PNG binary>, rooms_json: <JSON string>, changes: <JSON string> }
    Output: PNG binary
    Colors: red = illegal, yellow = requires_approval

    Strategy per change:
      - 2 rooms: find shared edge (tol=0.08) or closest edge pair + Hough snap
      - 1 room + hint_point: snap nearest Hough line to hint_point
      - fallback: draw room polygon outline
    """
    if 'image' not in request.files:
        return {'error': 'No image provided'}, 400
    for field in ('rooms_json', 'changes'):
        if field not in request.form:
            return {'error': f'{field} is required'}, 400

    img_bytes = request.files['image'].read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGBA')
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    rooms_raw = json.loads(request.form['rooms_json'])
    rooms = json.loads(rooms_raw) if isinstance(rooms_raw, str) else rooms_raw
    changes_raw = json.loads(request.form['changes'])
    changes = json.loads(changes_raw) if isinstance(changes_raw, str) else changes_raw
    width, height = img.size

    line_colors = {
        'illegal':           (220, 38,  38,  255),   # red
        'requires_approval': (217, 119, 6,   255),   # amber
    }
    label_bg = {
        'illegal':           (220, 38,  38,  220),
        'requires_approval': (217, 119, 6,   220),
    }

    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    badge_num = 1
    room_map = {r.get('id', ''): r for r in rooms}
    hough_radius = int(min(width, height) * 0.08)

    for change in changes:
        cls = change.get('classification', 'legal')
        if cls == 'legal':
            continue

        line_color = line_colors.get(cls, (150, 150, 150, 255))
        bg_color   = label_bg.get(cls, (150, 150, 150, 220))
        line_w = max(4, int(min(width, height) * 0.008))

        affected_ids = change.get('affected_room_ids') or [change.get('room_id', '')]
        hint_point = change.get('hint_point')  # [x, y] normalized, or None

        drawn_segment = None  # (x1, y1, x2, y2) pixels
        badge_pos = None

        if len(affected_ids) >= 2:
            # Wall between two rooms: try shared edge, then closest edge pair + Hough snap
            r1 = room_map.get(affected_ids[0], {})
            r2 = room_map.get(affected_ids[1], {})
            raw1 = region_to_polygon_raw(r1.get('polygon') or r1.get('region_percent', {}))
            raw2 = region_to_polygon_raw(r2.get('polygon') or r2.get('region_percent', {}))
            if raw1 and raw2:
                seg = find_shared_edge(raw1, raw2, width, height, tol=0.08)
                if seg:
                    drawn_segment = seg
                else:
                    pair = find_closest_edge_pair(raw1, raw2)
                    if pair:
                        a1, b1, a2, b2 = pair
                        cx = ((a1[0] + b1[0] + a2[0] + b2[0]) / 4) * width
                        cy = ((a1[1] + b1[1] + a2[1] + b2[1]) / 4) * height
                        snapped = find_nearest_hough_line(img_cv, cx, cy, hough_radius)
                        if snapped:
                            drawn_segment = snapped
                        else:
                            # Fallback: midline between the two edges
                            drawn_segment = (
                                int((a1[0] + a2[0]) / 2 * width),
                                int((a1[1] + a2[1]) / 2 * height),
                                int((b1[0] + b2[0]) / 2 * width),
                                int((b1[1] + b2[1]) / 2 * height),
                            )

        if drawn_segment is None and hint_point:
            # Single room with hint: snap to nearest Hough line near hint_point
            hx = hint_point[0] * width
            hy = hint_point[1] * height
            snapped = find_nearest_hough_line(img_cv, hx, hy, hough_radius)
            if snapped:
                drawn_segment = snapped

        if drawn_segment is not None:
            x1s, y1s, x2s, y2s = drawn_segment
            draw.line([(x1s, y1s), (x2s, y2s)], fill=line_color, width=line_w * 2)
            badge_pos = ((x1s + x2s) // 2, (y1s + y2s) // 2)
        else:
            # Fallback: draw room polygon outline
            for room_id in affected_ids:
                room = room_map.get(room_id, {})
                poly = region_to_polygon(
                    room.get('polygon') or room.get('region_percent', {}), width, height
                )
                if not poly:
                    continue
                for i in range(len(poly)):
                    draw.line([poly[i], poly[(i + 1) % len(poly)]], fill=line_color, width=line_w)
                if badge_pos is None:
                    badge_pos = polygon_centroid(poly)

        if badge_pos:
            mx, my = badge_pos
            r = line_w * 3
            draw.ellipse([mx - r, my - r, mx + r, my + r], fill=bg_color)
            draw.text((mx - r // 2, my - r), str(badge_num), fill=(255, 255, 255, 255))
            badge_num += 1

    result = Image.alpha_composite(img, overlay).convert('RGB')
    img_io = io.BytesIO()
    result.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png', download_name='annotated_changes.png')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
