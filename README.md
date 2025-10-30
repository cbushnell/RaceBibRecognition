# Race Bib Recognition

Automated race bib number detection and tagging system using computer vision and OCR.

## Features

- **OCR Detection**: Uses EasyOCR for text detection and bib number recognition
- **Face Recognition**: InsightFace for detecting and matching runners across multiple photos
- **Confidence Tracking**: Automatically updates bib numbers based on confidence scores
- **Metadata Tagging**: Writes bib numbers to IPTC keywords for easy photo organization
- **Retroactive Identification**: Identifies runners in photos without visible bibs using face matching
- **Automatic Cleanup**: Removes temporary artifact files after processing

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Process all images in a directory:

```bash
python main.py /path/to/race/photos
```

### Advanced Options

### Confidence Thresholds

**`--min-face-confidence`** (default: `0.8`)
- Minimum confidence score required for face detection

```bash
python main.py /path/to/photos --min-face-confidence 0.9
```

**`--min-bib-confidence`** (default: `0.5`)
- Minimum confidence score required for bib number detection

```bash
python main.py /path/to/photos --min-bib-confidence 0.7
```

#### Metadata Options

**`--no-metadata`**
- Skip writing bib numbers to image IPTC metadata

```bash
python main.py /path/to/photos --no-metadata
```

**`--overwrite-metadata`**
- Replace existing IPTC keywords instead of appending to them
- Default behavior is to append, preserving existing keywords

```bash
python main.py /path/to/photos --overwrite-metadata
```

**`--only-associated-bibs`**
- Only write bib numbers that are associated with detected faces
- Excludes standalone bib numbers found in the background
- Useful when you only want to tag bibs of people in the photo
```bash
python main.py /path/to/photos --only-associated-bibs
```

**`--bib-range`**
- Filter bib numbers to only include those within a specific range
- Format: `start-end` (e.g., `3000-4000`)
- Useful for races with multiple categories or divisions
```bash
python main.py /path/to/photos --bib-range 3000-4000
```

#### Combining Options

You can combine multiple options together:
```bash
python main.py /path/to/photos \
  --min-face-confidence 0.9 \
  --min-bib-confidence 0.7 \
  --only-associated-bibs \
  --bib-range 1000-2000 \
  --overwrite-metadata

### Help

```bash
python main.py --help
```

## Project Structure

```
RaceBibRecognition/
├── main.py                          # CLI entry point
├── src/
│   └── bib_recognition/
│       ├── __init__.py
│       ├── ocr.py                   # OCR and text detection using EasyOCR
│       ├── face_detection.py        # Face detection and association
│       ├── gallery.py               # Runner tracking across photos
│       ├── metadata.py              # IPTC metadata operations
│       └── processor.py             # Main processing pipeline
├── requirements.txt
└── README.md
```

## How It Works

1. **OCR Detection**: Scans each image for text regions and identifies bib numbers
2. **Face Detection**: Detects faces in the image using InsightFace
3. **Association**: Associates detected bib numbers with nearby faces based on spatial proximity
4. **Gallery Building**: Builds a gallery of known runners with their face embeddings
5. **Retroactive Identification**: Re-processes images to identify runners whose bibs weren't initially visible
6. **Metadata Writing**: Writes all detected bib numbers to IPTC keywords
7. **Cleanup**: Removes temporary backup files

## Output

- **Exit Code**: Returns 0 on success, 1 on error
- **Metadata**: Bib numbers written to IPTC keywords field
- **Statistics**: Prints summary of processed images, identified runners, and errors

## Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)
- Sufficient disk space for model weights (~2-3GB)

## Notes

- First run will download model weights from HuggingFace
- Processing time depends on image count and hardware
- Backup files (.bak) are created before modifying image metadata
- All detected bib numbers are tagged by default, even if not associated with a face

## License

See LICENSE file for details.
