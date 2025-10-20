"""
OCR module for detecting and refining bib numbers using Florence-2
"""

import re
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


class BibOCR:
    """Handles OCR operations for bib number detection"""

    def __init__(self, model_name="ducviet00/Florence-2-large-hf"):
        """
        Initialize the OCR model

        Args:
            model_name: HuggingFace model name for Florence-2
        """
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        print(f"Loading OCR model on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(model_name)
        self.model.to(self.device)

    def detect_text_regions(self, image):
        """
        Detect text regions in an image

        Args:
            image: PIL Image

        Returns:
            Dictionary with quad_boxes and labels
        """
        prompt = "<OCR_WITH_REGION>"

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            self.device, self.torch_dtype
        )

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=4096,
            num_beams=3,
            do_sample=False,
        )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        ocr_result = self.processor.post_process_generation(
            generated_text, task=prompt, image_size=(image.width, image.height)
        )

        return ocr_result.get('<OCR_WITH_REGION>', {'quad_boxes': [], 'labels': []})

    def refine_bib_number(self, image, bbox, original_number, scale_factor=4, min_size=200):
        """
        Refine a bib number reading by cropping and upscaling the region

        Args:
            image: PIL Image (original full image)
            bbox: [x, y, w, h] bounding box of the bib
            original_number: Original OCR result
            scale_factor: How much to upscale the cropped region (default: 4x)
            min_size: Minimum dimension for the cropped region (default: 200px)

        Returns:
            Tuple of (refined_bib_number, confidence_info) or (None, None) if refinement fails
        """
        try:
            x, y, w, h = bbox

            # Add padding around the bib (20% on each side)
            padding_x = int(w * 0.2)
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

            # Run OCR on the upscaled region
            prompt = "<OCR>"
            inputs = self.processor(text=prompt, images=upscaled, return_tensors="pt").to(
                self.device, self.torch_dtype
            )

            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
            )

            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            ocr_result = self.processor.post_process_generation(
                generated_text, task=prompt, image_size=(upscaled.width, upscaled.height)
            )

            # Extract just the numbers from the OCR result
            ocr_text = ocr_result.get('<OCR>', '')

            # Try to find the largest number in the text
            numbers = re.findall(r'\d+', ocr_text)

            if numbers:
                # Return the longest number (likely the bib number)
                bib_number = max(numbers, key=len)

                # If the refined number is longer than the original, use it
                if len(bib_number) > len(original_number):
                    confidence_info = {
                        'original_size': current_size,
                        'upscaled_size': max(new_width, new_height),
                        'ocr_text': ocr_text,
                        'all_numbers_found': numbers
                    }
                    return bib_number, confidence_info

                return original_number, {'original_size': current_size}

            return None, None

        except Exception as e:
            print(f"    Warning: Bib refinement failed: {e}")
            return None, None


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


def calculate_bib_confidence(original_number, refined_number, bbox_size, refinement_info):
    """
    Calculate a confidence score for a bib number detection

    Args:
        original_number: Original OCR result
        refined_number: Refined OCR result (after upscaling)
        bbox_size: Size of the bib bounding box in pixels
        refinement_info: Dictionary with refinement details (or None)

    Returns:
        confidence score (0.0 to 1.0)
    """
    confidence = 0.5  # Base confidence

    # Factor 1: Size (larger bibs are more reliable)
    size_score = min(bbox_size / 100.0, 1.0) * 0.3
    confidence += size_score

    # Factor 2: Consistency between original and refined
    if refined_number and original_number == refined_number:
        confidence += 0.2  # Consistent reading = higher confidence
    elif refined_number and original_number != refined_number:
        confidence -= 0.1  # Inconsistent = lower confidence

    # Factor 3: Number length (typical bib numbers are 3-4 digits)
    if refined_number:
        num_length = len(refined_number)
        if 3 <= num_length <= 5:
            confidence += 0.15  # Typical length
        elif num_length == 2:
            confidence -= 0.05  # Too short, might be partial
        elif num_length == 1:
            confidence -= 0.15  # Very short, likely wrong
        elif num_length > 5:
            confidence -= 0.1   # Too long, might include extra digits

    # Factor 4: Clean OCR text (fewer extraneous characters = better)
    if refinement_info and isinstance(refinement_info, dict) and 'ocr_text' in refinement_info:
        ocr_text = refinement_info['ocr_text']
        if refined_number and len(ocr_text.strip()) <= len(refined_number) + 3:
            confidence += 0.05

    # Clamp to 0.0-1.0 range
    return max(0.0, min(1.0, confidence))
