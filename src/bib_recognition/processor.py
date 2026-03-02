"""
Main processing pipeline for race bib recognition
"""

import os
from pathlib import Path
from PIL import Image

from .ocr import BibOCR, quad_to_bbox
from .face_detection import FaceDetector, associate_faces_with_bibs
from .gallery import RunnerGallery
from .metadata import write_bib_numbers_to_metadata, cleanup_artifacts


class BibRecognitionProcessor:
    """Main processor for race bib recognition"""

    def __init__(self, min_face_confidence=0.8, min_bib_confidence=0.5, bib_range=None):
        """
        Initialize the processor

        Args:
            min_face_confidence: Minimum confidence for face detection (default: 0.8)
            min_bib_confidence: Minimum confidence for bib OCR (default: 0.6)
            bib_range: Range of values for bibs to be considered for adding (in numbers)
        """
        self.face_detector = FaceDetector()
        self.gallery = RunnerGallery(threshold=0.4)
        self.min_face_confidence = min_face_confidence
        self.min_bib_confidence = min_bib_confidence
        self.ocr = BibOCR(bib_range=[int(bib_range.split("-")[0]), int(bib_range.split("-")[1])] if bib_range else None,
                          min_bib_confidence=self.min_bib_confidence)


    def process_image(self, image_path, refine_bibs=True, skip_gallery_ops=False):
        """
        Process a single image to detect faces and bib numbers

        Args:
            image_path: Path to the image file
            refine_bibs: Whether to refine bib numbers by upscaling (default: True)
            skip_gallery_ops: Skip gallery identification (used for batch processing, default: False)

        Returns:
            Dictionary with processing results
        """
        # Load image
        img = Image.open(image_path)

        # Run OCR to detect bib numbers
        ocr_result = self.ocr.detect_text_regions(img)

        # Detect faces
        faces = self.face_detector.detect_faces(img)

        # Extract and refine bib numbers from OCR results
        bib_boxes = []
        for quad_box, label, confidence in zip(ocr_result['quad_boxes'], ocr_result['labels'], ocr_result['confidences']):
            if label.isdigit():
                bbox = quad_to_bbox(quad_box)
                bib_size = max(bbox[2], bbox[3])

                bib_boxes.append({
                    'number': label,
                    'bbox': bbox,
                    'original_number': label,
                    'size': bib_size,
                    'confidence': confidence
                })

        # Refine bib numbers if requested
        if refine_bibs and len(bib_boxes) > 0:
            for bib in bib_boxes:
                refined_number = self.ocr.refine_bib_number(
                    img, bib['bbox'], bib['original_number'], bib['confidence']
                )

                if refined_number and refined_number != bib['original_number']:
                    bib['number'] = refined_number

        # Associate faces with bib numbers
        associations = associate_faces_with_bibs(
            faces,
            bib_boxes,
            img.width,
            img.height,
            self.min_face_confidence,
            self.min_bib_confidence
        )

        # Identify runners from gallery using batch operation (unless skipped for batch processing)
        if skip_gallery_ops:
            identifications = []
        else:
            embeddings = [assoc['face']['embedding'] for assoc in associations if assoc['face']['embedding'] is not None]

            if embeddings:
                # Batch identify all faces at once
                gallery_results = self.gallery.identify_runner(embeddings)

                # Build identifications with results
                identifications = []
                result_idx = 0
                for assoc in associations:
                    if assoc['face']['embedding'] is not None:
                        bib_from_gallery, distance = gallery_results[result_idx]
                        identifications.append({
                            'face_bbox': assoc['face']['bbox'],
                            'bib_from_ocr': assoc['bib_number'],
                            'bib_from_gallery': bib_from_gallery,
                            'match_distance': distance
                        })
                        result_idx += 1
            else:
                identifications = []

        return {
            'image': img,
            'image_path': image_path,
            'ocr_result': {'<OCR_WITH_REGION>': ocr_result},
            'associations': associations,
            'identifications': identifications
        }

    def process_directory(self, image_dir, write_metadata=True,
                         include_all_detected_bibs=True, overwrite_metadata=False):
        """
        Process all images in a directory

        Args:
            image_dir: Path to directory containing images
            write_metadata: Write bib numbers to IPTC metadata (default: True)
            include_all_detected_bibs: Include all detected bibs in metadata (default: True)
            overwrite_metadata: Overwrite existing metadata instead of appending (default: False)

        Returns:
            Dictionary with success status and statistics
        """
        # Find all image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
        image_files = []

        for file in os.listdir(image_dir):
            if file.endswith(image_extensions):
                image_files.append(os.path.join(image_dir, file))

        if not image_files:
            return {
                'success': False,
                'message': f"No image files found in {image_dir}",
                'images_processed': 0
            }

        print(f"Found {len(image_files)} image(s) to process")
        print("="*60)

        # Process all images (OCR + face detection only, no gallery operations yet)
        all_results = []
        errors = []

        for idx, img_path in enumerate(image_files, 1):
            try:
                print(f"\n[{idx}/{len(image_files)}] Processing {Path(img_path).name}...")
                result = self.process_image(img_path, refine_bibs=True, skip_gallery_ops=True)
                all_results.append(result)

            except Exception as e:
                error_msg = f"Error processing {Path(img_path).name}: {e}"
                print(f"  ✗ {error_msg}")
                errors.append(error_msg)

        # Batch gallery operations across ALL images
        print(f"\n{'='*60}")
        print("GALLERY OPERATIONS (BATCH)")
        print(f"{'='*60}")

        # Collect all associations from all images
        all_associations = []
        for result in all_results:
            all_associations.extend(result['associations'])

        # Add all associations to gallery in one batch
        if all_associations:
            print(f"Adding {len(all_associations)} face-bib associations to gallery...")
            self.gallery.add_associations(all_associations)

        # Retroactive identification pass - identify all unidentified faces in one batch
        if self.gallery.get_runner_count() > 0:
            print(f"\n{'='*60}")
            print("RETROACTIVE IDENTIFICATION PASS")
            print(f"{'='*60}")

            # Collect all unidentified faces across all results for batch processing
            unidentified_faces = []
            face_to_assoc_map = []  # Maps batch result index to (result_idx, assoc_idx)

            for result_idx, result in enumerate(all_results):
                for assoc_idx, assoc in enumerate(result['associations']):
                    if not assoc['bib_number'] and assoc['face']['embedding']:
                        unidentified_faces.append(assoc['face']['embedding'])
                        face_to_assoc_map.append((result_idx, assoc_idx))

            # Batch identify all unidentified faces at once
            if unidentified_faces:
                print(f"Identifying {len(unidentified_faces)} unidentified faces...")
                gallery_results = self.gallery.identify_runner(unidentified_faces)

                # Update associations with batch results
                matches_found = 0
                for i, (bib_from_gallery, distance) in enumerate(gallery_results):
                    if bib_from_gallery:
                        result_idx, assoc_idx = face_to_assoc_map[i]
                        assoc = all_results[result_idx]['associations'][assoc_idx]
                        assoc['bib_number'] = bib_from_gallery
                        assoc['retroactive_match'] = True
                        assoc['match_distance'] = distance
                        matches_found += 1

                print(f"  ✓ Found {matches_found} retroactive matches")
            else:
                print("  ✓ No unidentified faces to process")

        # Write metadata to images
        metadata_written = 0
        if write_metadata:
            print(f"\n{'='*60}")
            print("WRITING METADATA")
            print(f"{'='*60}")

            for result in all_results:
                bib_numbers = []

                if include_all_detected_bibs:
                    # Include ALL detected bib numbers from OCR
                    ocr_result = result.get('ocr_result', {})
                    if '<OCR_WITH_REGION>' in ocr_result:
                        for label in ocr_result['<OCR_WITH_REGION>'].get('labels', []):
                            if label.isdigit():
                                bib_numbers.append(label)

                # Also add bib numbers from associations
                for assoc in result['associations']:
                    if assoc['bib_number']:
                        if (assoc.get('confidence', {}).get('is_confident', False) or
                            assoc.get('retroactive_match')):
                            bib_numbers.append(assoc['bib_number'])

                # Remove duplicates
                bib_numbers = list(set(bib_numbers))

                if bib_numbers:
                    success = write_bib_numbers_to_metadata(
                        result['image_path'],
                        bib_numbers,
                        metadata_field='keywords',
                        backup=True,
                        overwrite=overwrite_metadata
                    )

                    if success:
                        print(f"  ✓ {Path(result['image_path']).name}: {', '.join(sorted(bib_numbers))}")
                        metadata_written += 1

        # Cleanup artifacts
        print(f"\n{'='*60}")
        print("CLEANUP")
        print(f"{'='*60}")
        deleted_count = cleanup_artifacts(image_dir)
        if deleted_count > 0:
            print(f"  ✓ Deleted {deleted_count} artifact file(s)")
        else:
            print(f"  ✓ No artifact files to clean up")

        # Final statistics
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Images processed: {len(all_results)}/{len(image_files)}")
        print(f"Metadata written: {metadata_written}")
        print(f"Known runners: {self.gallery.get_runner_count()}")
        print(f"Unknown runners: {self.gallery.get_unknown_runner_count()}")
        if errors:
            print(f"Errors: {len(errors)}")

        return {
            'success': len(errors) == 0,
            'images_processed': len(all_results),
            'images_total': len(image_files),
            'metadata_written': metadata_written,
            'runners_identified': self.gallery.get_runner_count(),
            'errors': errors
        }
