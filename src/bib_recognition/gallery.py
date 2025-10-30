"""
Runner gallery for tracking faces and bib numbers across multiple photos
"""

from scipy.spatial.distance import cosine, euclidean
from sklearn.neighbors import NearestNeighbors
import numpy as np


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

    def __init__(self, threshold=0.4, metric='cosine'):
        """
        Initialize the gallery

        Args:
            threshold: Distance threshold for face matching (default: 0.4)
            metric: Distance metric to use ('cosine' or 'euclidean', default: 'cosine')
        """
        self.gallery = {}  # bib_number -> list of embeddings
        self.bib_confidence = {}  # bib_number -> {'confidence': float, 'size': float, 'info': dict}
        self.unknown_runners = []  # List of embeddings for runners without bib numbers
        self.threshold = threshold
        self.next_unknown_id = 1
        self.metric = metric

        # KNN indices for fast lookup
        self.knn_gallery = None  # NearestNeighbors for known runners
        self.knn_unknown = None  # NearestNeighbors for unknown runners

        # Mappings to track which embedding belongs to which bib
        self.gallery_embeddings = []  # List of all gallery embeddings
        self.gallery_bib_map = []  # Parallel array: embedding index -> bib_number
        self.unknown_embeddings = []  # List of unknown representative embeddings
        self.unknown_id_map = []  # Parallel array: embedding index -> unknown_id

    def _rebuild_gallery_index(self):
        """
        Rebuild the KNN index for known runners (gallery).
        Called after adding/removing runners from the gallery.
        """
        self.gallery_embeddings = []
        self.gallery_bib_map = []

        for bib_number, embeddings in self.gallery.items():
            for emb in embeddings:
                self.gallery_embeddings.append(emb)
                self.gallery_bib_map.append(bib_number)

        if len(self.gallery_embeddings) > 0:
            self.knn_gallery = NearestNeighbors(
                n_neighbors=1,
                metric=self.metric,
                algorithm='auto'
            )
            self.knn_gallery.fit(np.array(self.gallery_embeddings))
        else:
            self.knn_gallery = None

    def _rebuild_unknown_index(self):
        """
        Rebuild the KNN index for unknown runners.
        Called after adding unknown runners.
        """
        self.unknown_embeddings = []
        self.unknown_id_map = []

        for unknown_runner in self.unknown_runners:
            self.unknown_embeddings.append(unknown_runner['representative_embedding'])
            self.unknown_id_map.append(unknown_runner['id'])

        if len(self.unknown_embeddings) > 0:
            self.knn_unknown = NearestNeighbors(
                n_neighbors=1,
                metric=self.metric,
                algorithm='auto'
            )
            self.knn_unknown.fit(np.array(self.unknown_embeddings))
        else:
            self.knn_unknown = None

    def add_runner(self, bib_number, face_embedding, confidence_info=None):
        """
        Add runner(s) face(s) to the gallery with confidence tracking.
        If a runner with this embedding already exists but with a different bib number,
        the bib number will be updated if the new one has higher confidence.

        Args:
            bib_number: The bib number detected, or list of bib numbers for batch operation
            face_embedding: Face embedding vector, or list of face embeddings for batch operation
            confidence_info: Dict with 'face', 'bib_size', 'bib_ocr', 'is_confident', 'bib_info'
                           or list of dicts for batch operation
        """
        # Detect batch vs single input
        is_batch = isinstance(bib_number, list)

        if is_batch:
            # Normalize inputs for batch processing
            bib_numbers = bib_number
            face_embeddings = face_embedding if isinstance(face_embedding, list) else [face_embedding] * len(bib_numbers)
            confidence_infos = confidence_info if isinstance(confidence_info, list) else [confidence_info] * len(bib_numbers)

            if len(bib_numbers) != len(face_embeddings):
                raise ValueError("bib_numbers and face_embeddings must have the same length")

            return self._add_runner_batch(bib_numbers, face_embeddings, confidence_infos)
        else:
            # Convert single to batch, process, return
            if face_embedding is None:
                return

            return self._add_runner_batch([bib_number], [face_embedding], [confidence_info])

    def _add_runner_batch(self, bib_numbers, face_embeddings, confidence_infos):
        """
        Internal batch processor for add_runner.
        Optimized to rebuild KNN index only once at the end.
        """
        # Check for existing faces in batch using current index
        existing_bibs = []
        if self.knn_gallery is not None and len(self.gallery_embeddings) > 0:
            valid_embeddings = [emb for emb in face_embeddings if emb is not None]
            if valid_embeddings:
                distances, indices = self.knn_gallery.kneighbors(np.array(valid_embeddings))

                valid_idx = 0
                for emb in face_embeddings:
                    if emb is None:
                        existing_bibs.append(None)
                    elif distances[valid_idx][0] < self.threshold:
                        existing_bibs.append(self.gallery_bib_map[indices[valid_idx][0]])
                        valid_idx += 1
                    else:
                        existing_bibs.append(None)
                        valid_idx += 1
            else:
                existing_bibs = [None] * len(face_embeddings)
        else:
            existing_bibs = [None] * len(face_embeddings)

        # Process each runner (without rebuilding index each time)
        for i, (bib_number, face_embedding, confidence_info) in enumerate(zip(bib_numbers, face_embeddings, confidence_infos)):
            if face_embedding is None:
                continue

            existing_bib = existing_bibs[i]

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

        # Rebuild the KNN index only once after all additions
        self._rebuild_gallery_index()

    def add_unknown_runner(self, face_embedding):
        """
        Add face(s) without a bib number.
        First check if it matches any existing unknown runner or known runner.

        Args:
            face_embedding: Face embedding vector, or list of face embeddings for batch operation

        Returns:
            Bib number or "UNKNOWN_X" identifier (single), or list of identifiers (batch)
        """
        # Detect batch vs single input
        is_batch = isinstance(face_embedding, list) and len(face_embedding) > 0 and isinstance(face_embedding[0], (list, np.ndarray))

        if is_batch:
            # Process batch
            return self._add_unknown_runner_batch(face_embedding)
        else:
            # Convert single to batch, process, and return single result
            if face_embedding is None:
                return None

            results = self._add_unknown_runner_batch([face_embedding])
            return results[0]

    def _add_unknown_runner_batch(self, face_embeddings):
        """
        Internal batch processor for add_unknown_runner.
        Optimized to rebuild KNN index only once at the end.
        """
        results = []

        # Check all embeddings against known runners in batch
        known_matches = self.identify_runner(face_embeddings)

        # Separate into known matches and truly unknown
        unknown_embeddings = []
        unknown_positions = []

        for i, (face_emb, (bib_match, _)) in enumerate(zip(face_embeddings, known_matches)):
            if face_emb is None:
                results.append(None)
            elif bib_match:
                results.append(bib_match)
            else:
                results.append(None)  # Placeholder
                unknown_embeddings.append(face_emb)
                unknown_positions.append(i)

        # Check unknown embeddings against existing unknown runners in batch
        if unknown_embeddings:
            if self.knn_unknown is not None and len(self.unknown_embeddings) > 0:
                distances, indices = self.knn_unknown.kneighbors(np.array(unknown_embeddings))

                for j, pos in enumerate(unknown_positions):
                    if distances[j][0] < self.threshold:
                        # Found matching unknown runner
                        unknown_id = self.unknown_id_map[indices[j][0]]
                        for unknown_runner in self.unknown_runners:
                            if unknown_runner['id'] == unknown_id:
                                unknown_runner['embeddings'].append(unknown_embeddings[j])
                                results[pos] = f"UNKNOWN_{unknown_id}"
                                break
                    else:
                        # New unknown runner
                        unknown_id = self.next_unknown_id
                        self.next_unknown_id += 1
                        self.unknown_runners.append({
                            'id': unknown_id,
                            'embeddings': [unknown_embeddings[j]],
                            'representative_embedding': unknown_embeddings[j]
                        })
                        results[pos] = f"UNKNOWN_{unknown_id}"
            else:
                # No existing unknowns, create new for all
                for j, pos in enumerate(unknown_positions):
                    unknown_id = self.next_unknown_id
                    self.next_unknown_id += 1
                    self.unknown_runners.append({
                        'id': unknown_id,
                        'embeddings': [unknown_embeddings[j]],
                        'representative_embedding': unknown_embeddings[j]
                    })
                    results[pos] = f"UNKNOWN_{unknown_id}"

            # Rebuild unknown index once
            self._rebuild_unknown_index()

        # Add known matches to gallery in batch
        known_bibs = [bib for bib, _ in known_matches if bib]
        known_embs = [emb for emb, (bib, _) in zip(face_embeddings, known_matches) if emb is not None and bib]

        if known_bibs:
            self.add_runner(known_bibs, known_embs)

        return results

    def add_associations(self, associations):
        """
        Add multiple face-bib associations to the gallery (optimized batch processing)

        Args:
            associations: List of association dictionaries
        """
        if not associations:
            return

        # Separate known runners from unknown runners
        known_bibs = []
        known_embeddings = []
        known_confidences = []

        unknown_embeddings = []

        for assoc in associations:
            if assoc['bib_number'] and assoc['face']['embedding']:
                known_bibs.append(assoc['bib_number'])
                known_embeddings.append(assoc['face']['embedding'])
                known_confidences.append(assoc.get('confidence'))
            elif assoc['face']['embedding']:
                unknown_embeddings.append(assoc['face']['embedding'])

        # Process in batches for better performance
        if known_bibs:
            self.add_runner(known_bibs, known_embeddings, known_confidences)

        if unknown_embeddings:
            self.add_unknown_runner(unknown_embeddings)

    def identify_runner(self, face_embedding):
        """
        Identify which runner this face belongs to (only checks known runners with bibs)

        Args:
            face_embedding: Face embedding vector or list of face embeddings

        Returns:
            If single embedding: (bib_number, confidence_distance) or (None, inf) if no match
            If multiple embeddings: List of (bib_number, confidence_distance) tuples
        """
        # Detect batch vs single input
        is_batch = isinstance(face_embedding, list) and len(face_embedding) > 0 and isinstance(face_embedding[0], (list, np.ndarray))

        if is_batch:
            # Process batch
            return self._identify_runner_batch(face_embedding)
        else:
            # Convert single to batch, process, and return single result
            if face_embedding is None:
                return None, float('inf')

            results = self._identify_runner_batch([face_embedding])
            return results[0]

    def _identify_runner_batch(self, face_embeddings):
        """
        Identify which runners multiple faces belong to (vectorized batch operation)

        Args:
            face_embeddings: List of face embedding vectors

        Returns:
            List of (bib_number, confidence_distance) tuples, one per input embedding
        """
        if not face_embeddings:
            return []

        # Filter out None embeddings but track their positions
        valid_indices = []
        valid_embeddings = []
        for i, emb in enumerate(face_embeddings):
            if emb is not None:
                valid_indices.append(i)
                valid_embeddings.append(emb)

        # Initialize results with (None, inf) for all positions
        results = [(None, float('inf'))] * len(face_embeddings)

        # Process valid embeddings in batch
        if valid_embeddings and self.knn_gallery is not None and len(self.gallery_embeddings) > 0:
            distances, indices = self.knn_gallery.kneighbors(np.array(valid_embeddings))

            for i, valid_idx in enumerate(valid_indices):
                best_distance = distances[i][0]
                if best_distance < self.threshold:
                    best_match = self.gallery_bib_map[indices[i][0]]
                    results[valid_idx] = (best_match, best_distance)

        return results

    def get_runner_count(self):
        """Get number of unique runners in gallery (with known bib numbers)"""
        return len(self.gallery)

    def get_unknown_runner_count(self):
        """Get number of unknown runners (without bib numbers)"""
        return len(self.unknown_runners)
