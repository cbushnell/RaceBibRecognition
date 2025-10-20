#!/usr/bin/env python3
"""
Race Bib Recognition - Main Entry Point

Process race photos to detect and tag bib numbers using OCR and face recognition.
"""

import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from bib_recognition.processor import BibRecognitionProcessor


def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(
        description='Process race photos to detect and tag bib numbers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/photos
  %(prog)s /path/to/photos --no-metadata
  %(prog)s /path/to/photos --min-face-confidence 0.9
        """
    )

    parser.add_argument(
        'directory',
        type=str,
        help='Directory containing race photos to process'
    )

    parser.add_argument(
        '--min-face-confidence',
        type=float,
        default=0.8,
        help='Minimum confidence for face detection (default: 0.8)'
    )

    parser.add_argument(
        '--min-bib-confidence',
        type=float,
        default=0.6,
        help='Minimum confidence for bib OCR (default: 0.6)'
    )

    parser.add_argument(
        '--no-metadata',
        action='store_true',
        help='Skip writing bib numbers to image metadata'
    )

    parser.add_argument(
        '--no-cleanup',
        action='store_true',
        help='Skip cleanup of artifact files (.bak, ~, .tmp)'
    )

    parser.add_argument(
        '--only-associated-bibs',
        action='store_true',
        help='Only write bib numbers associated with faces to metadata (exclude unassociated)'
    )

    args = parser.parse_args()

    # Validate directory
    image_dir = Path(args.directory)
    if not image_dir.exists():
        print(f"Error: Directory does not exist: {image_dir}")
        return 1

    if not image_dir.is_dir():
        print(f"Error: Path is not a directory: {image_dir}")
        return 1

    # Initialize processor
    print("Initializing Race Bib Recognition System...")
    print("="*60)
    processor = BibRecognitionProcessor(
        min_face_confidence=args.min_face_confidence,
        min_bib_confidence=args.min_bib_confidence
    )

    # Process directory
    try:
        result = processor.process_directory(
            str(image_dir),
            write_metadata=not args.no_metadata,
            cleanup=not args.no_cleanup,
            include_all_detected_bibs=not args.only_associated_bibs
        )

        # Print final result
        print("\n" + "="*60)
        if result['success']:
            print("✓ Processing completed successfully!")
            return 0
        else:
            print("⚠ Processing completed with errors")
            if result.get('errors'):
                print(f"\nErrors encountered:")
                for error in result['errors']:
                    print(f"  - {error}")
            return 1

    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        return 130

    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
