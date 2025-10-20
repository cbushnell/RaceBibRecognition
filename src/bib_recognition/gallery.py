"""
Runner gallery for tracking faces and bib numbers across multiple photos
"""

from scipy.spatial.distance import cosine, euclidean


def compare_faces(embedding1, embedding2, metric='cosine'):
    """
    Compare two face embeddings and return similarity score

    Args:
        embedding1: First face embedding vector
        embedding2: Second face embedding vector
        metric: 'cosine' or 'euclidean' distance

    Returns:
        distance: Lower values indicate more similar faces
    """
    if embedding1 is None or embedding2 is None:
        return float('inf')

    if metric == 'cosine':
        return cosine(embedding1, embedding2)
    else:
        return euclidean(embedding1, embedding2)


class RunnerGallery:
    """
    Manages a gallery of runner faces mapped to bib numbers.
    Tracks confidence scores for bib numbers to handle re-identification.
    """

    def __init__(self, threshold=0.4):
        """
        Initialize the gallery

        Args:
            threshold: Cosine distance threshold for face matching (default: 0.4)
        """
        self.gallery = {}  # bib_number -> list of embeddings
        self.bib_confidence = {}  # bib_number -> {'confidence': float, 'size': float, 'info': dict}
        self.unknown_runners = []  # List of embeddings for runners without bib numbers
        self.threshold = threshold
        self.next_unknown_id = 1

    def add_runner(self, bib_number, face_embedding, confidence_info=None):
        """
        Add a runner's face to the gallery with confidence tracking.
        If a runner with this embedding already exists but with a different bib number,
        the bib number will be updated if the new one has higher confidence.

        Args:
            bib_number: The bib number detected
            face_embedding: Face embedding vector
            confidence_info: Dict with 'face', 'bib_size', 'bib_ocr', 'is_confident', 'bib_info'
        """
        if face_embedding is None:
            return

        # Check if this face already exists under a different bib number
        existing_bib = None
        for existing_bib_num, embeddings in self.gallery.items():
            for emb in embeddings:
                distance = compare_faces(face_embedding, emb)
                if distance < self.threshold:
                    existing_bib = existing_bib_num
                    break
            if existing_bib:
                break

        # If found under different bib number, compare confidence
        if existing_bib and existing_bib != bib_number:
            existing_conf = self.bib_confidence.get(existing_bib, {}).get('confidence', 0.0)
            new_conf = confidence_info.get('bib_ocr', 0.0) if confidence_info else 0.0

            if new_conf > existing_conf:
                # Remove old bib number
                if existing_bib in self.gallery:
                    del self.gallery[existing_bib]
                if existing_bib in self.bib_confidence:
                    del self.bib_confidence[existing_bib]
            else:
                # Keep the existing bib number, add this embedding to it
                bib_number = existing_bib

        # Add to gallery
        if bib_number not in self.gallery:
            self.gallery[bib_number] = []

        self.gallery[bib_number].append(face_embedding)

        # Update confidence tracking if this is higher confidence
        if confidence_info:
            new_conf = confidence_info.get('bib_ocr', 0.0)
            existing_conf_data = self.bib_confidence.get(bib_number, {})
            existing_conf = existing_conf_data.get('confidence', 0.0)

            if new_conf > existing_conf:
                self.bib_confidence[bib_number] = {
                    'confidence': new_conf,
                    'size': confidence_info.get('bib_size', 0),
                    'info': confidence_info.get('bib_info')
                }

    def add_unknown_runner(self, face_embedding):
        """
        Add a face without a bib number.
        First check if it matches any existing unknown runner or known runner.

        Args:
            face_embedding: Face embedding vector

        Returns:
            Bib number or "UNKNOWN_X" identifier
        """
        if face_embedding is None:
            return None

        # Check if this face matches a known runner
        bib_match, distance = self.identify_runner(face_embedding)
        if bib_match:
            self.add_runner(bib_match, face_embedding)
            return bib_match

        # Check if it matches an existing unknown runner
        for unknown_runner in self.unknown_runners:
            distance = compare_faces(face_embedding, unknown_runner['representative_embedding'])
            if distance < self.threshold:
                unknown_runner['embeddings'].append(face_embedding)
                return f"UNKNOWN_{unknown_runner['id']}"

        # New unknown runner
        unknown_id = self.next_unknown_id
        self.next_unknown_id += 1
        self.unknown_runners.append({
            'id': unknown_id,
            'embeddings': [face_embedding],
            'representative_embedding': face_embedding
        })
        return f"UNKNOWN_{unknown_id}"

    def add_associations(self, associations):
        """
        Add multiple face-bib associations to the gallery

        Args:
            associations: List of association dictionaries
        """
        for assoc in associations:
            if assoc['bib_number'] and assoc['face']['embedding']:
                confidence_info = assoc.get('confidence')
                self.add_runner(assoc['bib_number'], assoc['face']['embedding'], confidence_info)
            elif assoc['face']['embedding']:
                self.add_unknown_runner(assoc['face']['embedding'])

    def identify_runner(self, face_embedding):
        """
        Identify which runner this face belongs to (only checks known runners with bibs)

        Args:
            face_embedding: Face embedding vector

        Returns:
            (bib_number, confidence_distance) or (None, inf) if no match
        """
        best_match = None
        best_distance = float('inf')

        for bib_number, embeddings in self.gallery.items():
            for gallery_embedding in embeddings:
                distance = compare_faces(face_embedding, gallery_embedding)

                if distance < best_distance and distance < self.threshold:
                    best_distance = distance
                    best_match = bib_number

        return best_match, best_distance

    def get_runner_count(self):
        """Get number of unique runners in gallery (with known bib numbers)"""
        return len(self.gallery)

    def get_unknown_runner_count(self):
        """Get number of unknown runners (without bib numbers)"""
        return len(self.unknown_runners)
