"""
IPTC metadata reading and writing for image files
"""

import os
from pathlib import Path


def write_bib_numbers_to_metadata(image_path, bib_numbers, metadata_field='keywords',
                                   backup=False, overwrite=False):
    """
    Write bib numbers to image IPTC metadata

    Args:
        image_path: Path to the image file
        bib_numbers: List of bib numbers to write (e.g., ['3278', '3289'])
        metadata_field: IPTC field to write to (default: 'keywords')
        backup: Create a backup before modifying (default: True)
        overwrite: Overwrite existing metadata in the field (default: False, appends instead)

    Returns:
        True if successful, False otherwise
    """
    try:
        from iptcinfo3 import IPTCInfo
        import shutil

        # Create backup if requested
        if backup:
            backup_path = f"{image_path}.bak"
            if not os.path.exists(backup_path):
                shutil.copy2(image_path, backup_path)

        # Load IPTC info
        info = IPTCInfo(image_path, force=True)

        # Prepare bib numbers as tags
        bib_tags = [str(bib) for bib in bib_numbers if bib]

        if not bib_tags:
            return False

        # Map metadata field names to IPTC field names
        field_mapping = {
            'keywords': 'keywords',
            'caption': 'caption/abstract',
            'subject': 'subject reference',
            'supplemental_category': 'supplemental category'
        }

        iptc_field = field_mapping.get(metadata_field, 'keywords')

        # Get existing data
        existing = info[iptc_field] if info[iptc_field] else []
        if overwrite:
            info[iptc_field] = bib_tags
        else:
            # Merge existing and new tags, removing duplicates
            new_values = list(set(existing + bib_tags))
            info[iptc_field] = new_values

        # Save the modified IPTC data
        info.save()

        return True

    except Exception as e:
        print(f"    Error writing metadata to {Path(image_path).name}: {e}")
        return False


def read_bib_numbers_from_metadata(image_path, metadata_field='keywords'):
    """
    Read bib numbers from image IPTC metadata

    Args:
        image_path: Path to the image file
        metadata_field: IPTC field to read from (default: 'keywords')

    Returns:
        List of bib numbers found
    """
    try:
        from iptcinfo3 import IPTCInfo

        info = IPTCInfo(image_path, force=True)

        field_mapping = {
            'keywords': 'keywords',
            'caption': 'caption/abstract',
            'subject': 'subject reference',
            'supplemental_category': 'supplemental category'
        }

        iptc_field = field_mapping.get(metadata_field, 'keywords')

        # Get data from field
        if iptc_field == 'keywords':
            data = info['keywords'] if info['keywords'] else []
        elif iptc_field == 'caption/abstract':
            data = [info['caption/abstract']] if info['caption/abstract'] else []
        else:
            data = info[iptc_field] if info[iptc_field] else []

        # Extract bib numbers
        bib_numbers = []
        for item in data:
            item_str = str(item) if not isinstance(item, str) else item
            # Check if it's a number 
            if item_str.isdigit():
                # New format - just the number
                bib_numbers.append(item_str)

        return bib_numbers

    except Exception as e:
        print(f"    Error reading metadata from {Path(image_path).name}: {e}")
        return []


def cleanup_artifacts(image_dir):
    """
    Clean up artifact files (.bak, ~, .tmp) from the image directory

    Args:
        image_dir: Path to the images directory

    Returns:
        Number of files deleted
    """
    import glob

    artifact_patterns = ['*.bak', '*~', '*.tmp']
    deleted_count = 0

    for pattern in artifact_patterns:
        full_pattern = os.path.join(image_dir, pattern)
        for file_path in glob.glob(full_pattern):
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception:
                    pass

    return deleted_count
