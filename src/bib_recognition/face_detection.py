"""
Face detection and embedding extraction using InsightFace
"""

import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis


class FaceDetector:
    """Handles face detection and embedding extraction"""

    def __init__(self, det_size=(640, 640)):
        """
        Initialize the face detection model

        Args:
            det_size: Detection size tuple (default: (640, 640))
        """
        print("Loading face detection model...")
        self.face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=det_size)

    def detect_faces(self, image):
        """
        Detect faces in an image and extract their embeddings

        Args:
            image: PIL Image or image path

        Returns:
            List of dictionaries containing face info:
            - bbox: [x, y, w, h] bounding box
            - embedding: face embedding vector (512-dimensional)
            - confidence: detection confidence
        """
        try:
            # Convert PIL Image to numpy array
            if isinstance(image, Image.Image):
                img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                img = cv2.imread(image)

            # Detect faces and extract embeddings
            faces = self.face_app.get(img)

            results = []
            for face in faces:
                # Get bounding box - InsightFace returns [x1, y1, x2, y2]
                bbox_xyxy = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox_xyxy

                # Convert to [x, y, w, h] format
                x = x1
                y = y1
                w = x2 - x1
                h = y2 - y1

                # Get embedding (512-dimensional vector)
                embedding = face.embedding.tolist()

                # Get detection confidence
                confidence = float(face.det_score)

                results.append({
                    'bbox': [x, y, w, h],
                    'confidence': confidence,
                    'embedding': embedding
                })

            return results

        except Exception as e:
            print(f"Error detecting faces: {e}")
            import traceback
            traceback.print_exc()
            return []


def associate_faces_with_bibs(faces, bib_boxes, image_width, image_height,
                               min_face_confidence=0.8, min_bib_confidence=0.6):
    """
    Associate detected faces with bib numbers based on spatial proximity

    Args:
        faces: List of face dictionaries from detect_faces
        bib_boxes: List of bib box dictionaries with number, bbox, size, confidence_score
        image_width: Width of the image
        image_height: Height of the image
        min_face_confidence: Minimum confidence for face detection (default: 0.8)
        min_bib_confidence: Minimum confidence for bib OCR (default: 0.6)

    Returns:
        List of associations: [{'face': face_dict, 'bib_number': str, 'bib_bbox': [...], 'confidence': dict}]
    """
    associations = []

    # Filter faces by confidence
    confident_faces = [f for f in faces if f['confidence'] >= min_face_confidence]
    low_confidence_faces = [f for f in faces if f['confidence'] < min_face_confidence]

    if low_confidence_faces:
        print(f"  Filtered out {len(low_confidence_faces)} low-confidence face(s)")

    print(f"  Found {len(confident_faces)} confident faces and {len(bib_boxes)} bib numbers")

    # Track which bibs have been assigned
    assigned_bibs = set()

    # For each face, find the closest unassigned bib number
    for face_idx, face in enumerate(confident_faces):
        face_x, face_y, face_w, face_h = face['bbox']
        face_center_x = face_x + face_w / 2
        face_center_y = face_y + face_h / 2
        face_bottom = face_y + face_h

        best_bib = None
        best_bib_idx = None
        best_distance = float('inf')
        max_search_distance = image_height * 0.5

        for bib_idx, bib in enumerate(bib_boxes):
            # Skip if already assigned to another face
            if bib_idx in assigned_bibs:
                continue

            # Skip if bib confidence is too low
            if bib.get('confidence_score') and bib['confidence_score'] < min_bib_confidence:
                continue

            bib_x, bib_y, bib_w, bib_h = bib['bbox']
            bib_center_x = bib_x + bib_w / 2
            bib_center_y = bib_y + bib_h / 2

            # Calculate distances
            horizontal_distance = abs(face_center_x - bib_center_x)
            vertical_distance = bib_center_y - face_bottom

            # Bib should be reasonably below the face
            if vertical_distance > -face_h * 0.3:
                distance = horizontal_distance * 3 + abs(vertical_distance) * 0.5

                if distance < max_search_distance and distance < best_distance:
                    if horizontal_distance < face_w * 1.5:
                        best_distance = distance
                        best_bib = bib
                        best_bib_idx = bib_idx

        # Only assign if we found a reasonable match
        if best_bib and best_distance < max_search_distance:
            assigned_bibs.add(best_bib_idx)

            bib_conf = best_bib.get('confidence_score', 0.5)
            is_confident = (face['confidence'] >= min_face_confidence and
                          bib_conf >= min_bib_confidence)

            associations.append({
                'face': face,
                'bib_number': best_bib['number'],
                'bib_bbox': best_bib['bbox'],
                'match_distance': best_distance,
                'confidence': {
                    'face': face['confidence'],
                    'bib_size': best_bib.get('size', 0),
                    'bib_ocr': bib_conf,
                    'is_confident': is_confident,
                    'bib_info': best_bib.get('confidence_info')
                }
            })
        else:
            # Face detected but no bib number found or assigned
            associations.append({
                'face': face,
                'bib_number': None,
                'bib_bbox': None,
                'match_distance': None,
                'confidence': {
                    'face': face['confidence'],
                    'bib_size': None,
                    'bib_ocr': None,
                    'is_confident': False
                }
            })

    return associations
