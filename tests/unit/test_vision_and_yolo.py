import cv2
import numpy as np

from app.server import decode_array, readiness, terrain_payload
from engine.search_rescue import (AerialDetection, DetectionBand, FrameMetadata,
                                  GeoPoint, SearchAndRescueMission, SearchMission)
from engine.search_rescue.models import SearchCell
from engine.search_rescue.yolo_detector import YoloRgbDetector
from engine.search_rescue.vision import prepare_rgb


class _Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _Boxes:
    xyxy = _Tensor([[-5, -4, 18, 20], [2, 2, 10, 10]])
    conf = _Tensor([.91, .99])
    cls = _Tensor([0, 1])


class _Result:
    names = {0: "person", 1: "car"}
    boxes = _Boxes()


class _Model:
    def predict(self, **kwargs):
        assert kwargs["source"].dtype == np.uint8
        assert kwargs["source"].shape == (24, 32, 3)
        return [_Result()]


def test_opencv_decode_and_rgb_normalization():
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    image[..., 1] = 255
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    assert ok
    decoded = decode_array(encoded.tobytes(), "frame.png")
    assert decoded.shape == image.shape
    assert np.array_equal(prepare_rgb(decoded), image)


def test_yolo_filters_person_and_clips_boxes():
    detector = YoloRgbDetector("mock.pt")
    detector.model = _Model()
    metadata = FrameMetadata("frame", GeoPoint(30.0, 77.0), image_width=32, image_height=24)
    detections = detector.detect(np.zeros((24, 32), dtype=np.uint16), metadata)
    assert len(detections) == 1
    assert detections[0].label == "person"
    assert detections[0].band.value == "rgb"


def test_readiness_and_terrain_payload_are_explicit():
    status = readiness()
    assert "opencv_available" in status
    assert "fallback_reason" in status
    assert terrain_payload()["source"] in {"dem", "synthetic"}


def test_field_packet_preserves_detection_sensor_metadata():
    home = GeoPoint(30.0, 77.0, 1200)
    mission = SearchAndRescueMission(SearchMission("m", home, [SearchCell("A", home, 1000)]))
    mission.replan([AerialDetection("d", home, "person", .8, DetectionBand.THERMAL, "thermal-1", "hot region")])
    detection = mission.field_packet()["detections"][0]
    assert detection["band"] == "thermal"
    assert detection["source_frame"] == "thermal-1"
