"""
OCR module for detecting and refining bib numbers using EasyOCR
"""

import re
import easyocr
from PIL import Image
import numpy as np


class BibOCR:
    """Handles OCR operations for bib number detection using EasyOCR"""

    def __init__(self, languages=['en'], gpu=True, bib_range=None, min_bib_confidence=0.5):
        """
        Initialize the OCR model

        Args:
            languages: List of language codes for EasyOCR (default: ['en'])
            gpu: Whether to use GPU acceleration if available (default: True)
        """
        print(f"Loading EasyOCR model with languages: {languages}...")
        self.reader = easyocr.Reader(languages, gpu=gpu)
        print("EasyOCR model loaded successfully")
        self.min_bib_confidence = min_bib_confidence
        print(f"Minimum OCR confidence set to {self.min_bib_confidence}")
        self.bib_range = bib_range
        if self.bib_range:
            print(f"Only considering bibs in the range {self.bib_range[0]} to {self.bib_range[1]}")
            

    def detect_text_regions(self, image):
        """
        Detect text regions in an image

        Args:
            image: PIL Image

        Returns:
            Dictionary with quad_boxes, labels, and confidences (compatible with Florence-2 format)
        """
        # Convert PIL Image to numpy array
        img_array = np.array(image)

        # Run EasyOCR detection
        results = self.reader.readtext(img_array)

        # Convert EasyOCR results to Florence-2 compatible format
        quad_boxes = []
        labels = []
        confidences = []

        for detection in results:
            bbox, text, confidence = detection
            try:
                if (
                    (self.bib_range and int(text) in range(self.bib_range[0], self.bib_range[1] + 1)) or not self.bib_range
                    and confidence >= self.min_bib_confidence
                ):
                    # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    # Flatten to [x1, y1, x2, y2, x3, y3, x4, y4]
                    quad_box = [coord for point in bbox for coord in point]
                    quad_boxes.append(quad_box)
                    labels.append(text)
                    confidences.append(confidence)
            except:
                continue

        return {'quad_boxes': quad_boxes, 'labels': labels, 'confidences': confidences}

    def refine_bib_number(self, image, bbox, original_number, confidence, scale_factor=4, min_size=200):
        """
        Refine a bib number reading by cropping and upscaling the region

        Args:
            image: PIL Image (original full image)
            bbox: [x, y, w, h] bounding box of the bib
            original_number: Original OCR result
            confidence: Confidence of model's OCR prediction
            scale_factor: How much to upscale the cropped region (default: 4x)
            min_size: Minimum dimension for the cropped region (default: 200px)

        Returns:
            refined_bib_number
        """
        try:
            x, y, w, h = bbox

            # Add padding around the bib
            padding_x = int(w * 0.4)
            padding_y = int(h * 0.2)

            # Calculate padded crop region
            crop_x1 = max(0, x - padding_x)
            crop_y1 = max(0, y - padding_y)
            crop_x2 = min(image.width, x + w + padding_x)
            crop_y2 = min(image.height, y + h + padding_y)

            # Crop the bib region
            cropped = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

            # Calculate upscale factor based on size
            current_size = max(cropped.width, cropped.height)
            if current_size < min_size:
                adaptive_scale = min_size / current_size
            else:
                adaptive_scale = scale_factor

            # Upscale the cropped region
            new_width = int(cropped.width * adaptive_scale)
            new_height = int(cropped.height * adaptive_scale)
            upscaled = cropped.resize((new_width, new_height), Image.LANCZOS)

            # Convert to numpy array for EasyOCR
            upscaled_array = np.array(upscaled)

            # Run OCR on the upscaled region
            results = self.reader.readtext(upscaled_array)

            # Find all detected bib numbers matching defined criteria
            numbers = [{'bbox': bbox, 'text': text, "conf": conf} for (bbox, text, conf) in results if self.bib_range and text.isnumeric() and int(text) in range(self.bib_range[0], self.bib_range[1] + 1)]

            if numbers:
                max_confidence_number = max(numbers, key=lambda num: num['conf'])

                # If the model is more confident in the new detection, return it
                if max_confidence_number['conf'] > confidence:
                    return max_confidence_number['text']

                return original_number

            return None

        except Exception as e:
            print(f"    Warning: Bib refinement failed: {e}")
            return None


def quad_to_bbox(quad_box):
    """
    Convert quad box [x1, y1, x2, y2, x3, y3, x4, y4] to bbox [x, y, w, h]

    Args:
        quad_box: List of 8 coordinates

    Returns:
        [x, y, w, h] bounding box
    """
    x_coords = [quad_box[i] for i in range(0, 8, 2)]
    y_coords = [quad_box[i] for i in range(1, 8, 2)]

    x = min(x_coords)
    y = min(y_coords)
    w = max(x_coords) - x
    h = max(y_coords) - y

    return [x, y, w, h]