"""Lightweight local shim for `face_recognition` to allow running the
app without the heavy `dlib` dependency.

IMPORTANT: This is a mock implementation for development/testing only.
It does not perform real face recognition — it returns no encodings and
neutral distances. Replace with the real `face_recognition` package for
production.
"""
import numpy as np


def face_encodings(image, known_face_locations=None):
    """Return an empty list (no faces detected).

    The real `face_recognition.face_encodings` returns a list of 128-d
    face descriptors. This shim returns an empty list so code paths
    that handle "no face detected" will run.
    """
    return []


def compare_faces(known_encodings, face_encoding_to_check, tolerance=0.6):
    """Compare known encodings to a given encoding and return boolean list.

    If `known_encodings` is empty, returns an empty list.
    """
    if not known_encodings:
        return []
    dists = face_distance(known_encodings, face_encoding_to_check)
    return list(dists <= tolerance)


def face_distance(known_encodings, face_encoding_to_check):
    """Compute euclidean distances between known encodings and the given one.

    The shim assumes encodings are sequences of numbers. If encodings are
    empty, returns an empty numpy array.
    """
    if not known_encodings:
        return np.array([])
    ke = np.array(known_encodings, dtype=float)
    fe = np.array(face_encoding_to_check, dtype=float)
    # Broadcast subtraction and compute L2 norm along last axis
    try:
        dif = ke - fe
        return np.linalg.norm(dif, axis=1)
    except Exception:
        # If shapes are incompatible, return large distances
        return np.full((len(known_encodings),), 1e6)
