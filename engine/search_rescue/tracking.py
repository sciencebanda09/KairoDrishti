"""Candidate-track association with an honest, offline-safe fallback.

Mahalanobis gating is always active.  IMM is available behind the
``TRACKING_MODE`` flag, but the default is ``nearest_decay`` because a
last-observation tracker is easier to validate for field planning and never
pretends that a motion estimate is ground truth.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from math import exp, pi, sqrt

import numpy as np

from .models import AerialDetection, GeoPoint


TRACKING_MODE = os.environ.get("KAIRODRISHTI_TRACKING_MODE", "nearest_decay").lower()
TRACKING_MODES = {"imm", "nearest_decay"}
MAHALANOBIS_CONFIDENCE = 0.95
# Chi-square 95% quantile for two position degrees of freedom.
MAHALANOBIS_CHI2_THRESHOLD_95 = 5.991464547107979
DEFAULT_POSITION_COVARIANCE = np.diag([0.00018 ** 2, 0.00018 ** 2])
MEASUREMENT_COVARIANCE = np.diag([0.00008 ** 2, 0.00008 ** 2])
NEAREST_DECAY_RATE_PER_MINUTE = 0.12
IMM_MODE_NAMES = ("cv", "ct")
SIMULATION_TRACKING_FRAMING = (
    "SIMULATION / PLANNING ONLY: a track is an investigation cue, never "
    "confirmation of a survivor; KairoDrishti does not control aircraft or "
    "claim sensor-fused ground truth."
)


@dataclass
class Track:
    """A candidate track in ``[lat, lon, lat_per_min, lon_per_min]`` units."""

    track_id: str
    state: np.ndarray
    covariance: np.ndarray
    detections: list[AerialDetection] = field(default_factory=list)
    model_probabilities: dict[str, float] = field(
        default_factory=lambda: {"cv": .5, "ct": .5}
    )
    last_updated_minute: float = 0.0
    confidence: float = 0.0
    base_confidence: float = 0.0
    model_states: dict[str, np.ndarray] = field(default_factory=dict)
    model_covariances: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.state = np.asarray(self.state, dtype=float).reshape(4)
        self.covariance = np.asarray(self.covariance, dtype=float).reshape(4, 4)

    @property
    def position(self) -> GeoPoint:
        return GeoPoint(float(self.state[0]), float(self.state[1]))

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.state[2]), float(self.state[3])

    @property
    def latest_detection(self) -> AerialDetection | None:
        return self.detections[-1] if self.detections else None


def _measurement(detection: AerialDetection) -> np.ndarray:
    return np.asarray([detection.location.lat, detection.location.lon], dtype=float)


def mahalanobis_gate(track: Track, new_detection: AerialDetection,
                     position_covariance: np.ndarray | float) -> bool:
    """Return whether a detection is within the stated 95% 2-DOF gate."""
    covariance = np.asarray(position_covariance, dtype=float)
    if covariance.ndim == 0:
        covariance = np.eye(2) * float(covariance)
    elif covariance.ndim == 1:
        covariance = np.diag(covariance)
    covariance = covariance.reshape(2, 2)
    innovation = _measurement(new_detection) - track.state[:2]
    innovation_covariance = track.covariance[:2, :2] + covariance
    distance = float(innovation.T @ np.linalg.pinv(innovation_covariance) @ innovation)
    return distance <= MAHALANOBIS_CHI2_THRESHOLD_95


def _mahalanobis_distance(track: Track, detection: AerialDetection) -> float:
    innovation = _measurement(detection) - track.state[:2]
    covariance = track.covariance[:2, :2] + DEFAULT_POSITION_COVARIANCE
    return float(innovation.T @ np.linalg.pinv(covariance) @ innovation)


def associate_detections(
    tracks: list[Track], new_detections: list[AerialDetection]
) -> tuple[list[tuple[Track, AerialDetection]], list[AerialDetection]]:
    """Gate every pair, then greedily resolve a one-to-one nearest match."""
    candidates: list[tuple[float, int, int]] = []
    for track_index, track in enumerate(tracks):
        for detection_index, detection in enumerate(new_detections):
            if mahalanobis_gate(track, detection, DEFAULT_POSITION_COVARIANCE):
                candidates.append((_mahalanobis_distance(track, detection),
                                   track_index, detection_index))
    matches: list[tuple[Track, AerialDetection]] = []
    used_tracks: set[int] = set()
    used_detections: set[int] = set()
    for _, track_index, detection_index in sorted(candidates):
        if track_index in used_tracks or detection_index in used_detections:
            continue
        used_tracks.add(track_index)
        used_detections.add(detection_index)
        matches.append((tracks[track_index], new_detections[detection_index]))
    unmatched = [detection for index, detection in enumerate(new_detections)
                 if index not in used_detections]
    return matches, unmatched


def _cv_transition(dt: float) -> np.ndarray:
    return np.asarray([[1, 0, dt, 0], [0, 1, 0, dt],
                       [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)


def _ct_transition(dt: float, turn_rate: float = .035) -> np.ndarray:
    theta = turn_rate * dt
    if abs(turn_rate) < 1e-7:
        return _cv_transition(dt)
    s, c = np.sin(theta), np.cos(theta)
    return np.asarray([
        [1, 0, s / turn_rate, -(1 - c) / turn_rate],
        [0, 1, (1 - c) / turn_rate, s / turn_rate],
        [0, 0, c, -s],
        [0, 0, s, c],
    ], dtype=float)


def _process_covariance(dt: float) -> np.ndarray:
    q = max(dt, .001) ** 2 * 1e-8
    return np.diag([q, q, q * 10, q * 10])


def _ensure_imm(track: Track) -> None:
    if not track.model_states:
        track.model_states = {name: track.state.copy() for name in IMM_MODE_NAMES}
    if not track.model_covariances:
        track.model_covariances = {name: track.covariance.copy() for name in IMM_MODE_NAMES}
    probabilities = {name: float(track.model_probabilities.get(name, .5)) for name in IMM_MODE_NAMES}
    total = sum(probabilities.values()) or 1.0
    track.model_probabilities = {name: value / total for name, value in probabilities.items()}


def _combine_models(track: Track) -> None:
    _ensure_imm(track)
    state = sum((track.model_probabilities[name] * track.model_states[name]
                 for name in IMM_MODE_NAMES), np.zeros(4))
    covariance = np.zeros((4, 4))
    for name in IMM_MODE_NAMES:
        delta = track.model_states[name] - state
        covariance += track.model_probabilities[name] * (
            track.model_covariances[name] + np.outer(delta, delta))
    track.state, track.covariance = state, covariance


def imm_predict(track: Track, dt: float) -> Track:
    """Predict both constant-velocity and constant-turn-rate models."""
    _ensure_imm(track)
    dt = max(float(dt), 0.0)
    if dt == 0:
        return track
    transition = np.asarray([[.93, .07], [.10, .90]], dtype=float)
    prior = np.asarray([track.model_probabilities[name] for name in IMM_MODE_NAMES])
    predicted_probabilities = prior @ transition
    mixed_states: dict[str, np.ndarray] = {}
    mixed_covariances: dict[str, np.ndarray] = {}
    for target_index, target_name in enumerate(IMM_MODE_NAMES):
        weights = prior * transition[:, target_index]
        weights /= max(predicted_probabilities[target_index], 1e-12)
        mixed_state = sum((weights[source_index] * track.model_states[source_name]
                           for source_index, source_name in enumerate(IMM_MODE_NAMES)), np.zeros(4))
        mixed_covariance = np.zeros((4, 4))
        for source_index, source_name in enumerate(IMM_MODE_NAMES):
            delta = track.model_states[source_name] - mixed_state
            mixed_covariance += weights[source_index] * (
                track.model_covariances[source_name] + np.outer(delta, delta))
        mixed_states[target_name] = mixed_state
        mixed_covariances[target_name] = mixed_covariance
    for name in IMM_MODE_NAMES:
        transition_matrix = _cv_transition(dt) if name == "cv" else _ct_transition(dt)
        mixed_state = mixed_states[name]
        mixed_covariance = mixed_covariances[name]
        track.model_states[name] = transition_matrix @ mixed_state
        track.model_covariances[name] = (
            transition_matrix @ mixed_covariance @ transition_matrix.T + _process_covariance(dt))
    track.model_probabilities = {name: float(predicted_probabilities[index])
                                 for index, name in enumerate(IMM_MODE_NAMES)}
    _combine_models(track)
    return track


def imm_update(track: Track, detection: AerialDetection) -> Track:
    """Perform a linear position update for each IMM model and recombine."""
    _ensure_imm(track)
    H = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    measurement = _measurement(detection)
    likelihoods: dict[str, float] = {}
    for name in IMM_MODE_NAMES:
        state = track.model_states[name]
        covariance = track.model_covariances[name]
        innovation = measurement - H @ state
        innovation_covariance = H @ covariance @ H.T + MEASUREMENT_COVARIANCE
        gain = covariance @ H.T @ np.linalg.pinv(innovation_covariance)
        updated_state = state + gain @ innovation
        identity = np.eye(4)
        updated_covariance = ((identity - gain @ H) @ covariance @
                              (identity - gain @ H).T +
                              gain @ MEASUREMENT_COVARIANCE @ gain.T)
        track.model_states[name] = updated_state
        track.model_covariances[name] = updated_covariance
        determinant = max(float(np.linalg.det(innovation_covariance)), 1e-24)
        likelihoods[name] = exp(-.5 * float(innovation.T @ np.linalg.pinv(innovation_covariance) @ innovation)) / (
            2 * pi * sqrt(determinant))
    posterior = np.asarray([track.model_probabilities[name] * likelihoods[name]
                            for name in IMM_MODE_NAMES])
    posterior /= max(float(posterior.sum()), 1e-24)
    track.model_probabilities = {name: float(posterior[index])
                                 for index, name in enumerate(IMM_MODE_NAMES)}
    _combine_models(track)
    return track


def _new_track(track_id: str, detection: AerialDetection, minute: float) -> Track:
    state = np.asarray([detection.location.lat, detection.location.lon, 0.0, 0.0])
    covariance = np.diag([0.00018 ** 2, 0.00018 ** 2, .001 ** 2, .001 ** 2])
    detection.track_id = track_id
    return Track(track_id, state, covariance, [detection],
                 last_updated_minute=minute,
                 confidence=detection.confidence,
                 base_confidence=detection.confidence)


class DetectionTracker:
    """Mission-facing tracker with a complete nearest-decay fallback."""

    def __init__(self, mode: str | None = None) -> None:
        selected = (mode or TRACKING_MODE).lower()
        if selected not in TRACKING_MODES:
            raise ValueError(f"unsupported tracking mode: {selected}")
        self.mode = selected
        self.tracks: list[Track] = []
        self._next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1

    def ingest(self, detections: list[AerialDetection], minute: float = 0.0) -> list[Track]:
        matches, unmatched = associate_detections(self.tracks, detections)
        for track, detection in matches:
            previous = track.latest_detection
            elapsed = max(0.0, minute - track.last_updated_minute)
            detection.track_id = track.track_id
            if previous is not None:
                # A track keeps a stable first detection ID for mission
                # investigation bookkeeping while its track_id remains the
                # canonical identity.
                detection.detection_id = previous.detection_id
            if self.mode == "imm":
                imm_predict(track, elapsed)
                imm_update(track, detection)
            else:
                track.state[:2] = _measurement(detection)
                track.state[2:] = 0.0
            track.detections.append(detection)
            track.last_updated_minute = minute
            track.confidence = max(0.0, min(1.0, detection.confidence))
            track.base_confidence = track.confidence
        for detection in unmatched:
            track_id = f"TRK-{self._next_id:03d}"
            self._next_id += 1
            self.tracks.append(_new_track(track_id, detection, minute))
        return self.tracks

    def advance(self, minute: float) -> None:
        for track in self.tracks:
            elapsed = max(0.0, float(minute) - track.last_updated_minute)
            if self.mode == "nearest_decay":
                track.confidence = max(0.0, track.base_confidence *
                                       exp(-NEAREST_DECAY_RATE_PER_MINUTE * elapsed))
                if track.latest_detection is not None:
                    track.state[:2] = _measurement(track.latest_detection)
                    track.state[2:] = 0.0
            else:
                imm_predict(track, elapsed)

    def latest_detections(self) -> list[AerialDetection]:
        result = []
        for track in self.tracks:
            detection = track.latest_detection
            if detection is None:
                continue
            detection.track_id = track.track_id
            detection.confidence = max(0.0, min(1.0, track.confidence))
            result.append(detection)
        return result

    def packet_tracks(self) -> list[dict]:
        return [{
            "track_id": track.track_id,
            "lat": round(float(track.state[0]), 7),
            "lon": round(float(track.state[1]), 7),
            "velocity": {"lat_per_minute": round(float(track.state[2]), 8),
                         "lon_per_minute": round(float(track.state[3]), 8)},
            "confidence": round(max(0.0, min(1.0, track.confidence)), 4),
            "detection_count": len(track.detections),
            "last_updated_minute": round(track.last_updated_minute, 4),
            "tracking_mode": self.mode,
        } for track in self.tracks]
