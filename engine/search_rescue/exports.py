"""Portable field-packet exporters."""
from __future__ import annotations

import csv
import io
from html import escape
from typing import Any


WAYPOINT_FIELDS = (
    "mission_id", "record_type", "sequence", "kind", "latitude", "longitude",
    "altitude_m", "dwell_seconds", "cell_id", "sensor", "source_frame",
    "confidence", "rationale", "timestamp", "crs",
)


def packet_records(packet: dict[str, Any]) -> list[dict[str, Any]]:
    mission_id = packet.get("mission_id", "")
    records: list[dict[str, Any]] = []
    for waypoint in packet.get("waypoints", []):
        records.append({
            "mission_id": mission_id, "record_type": "waypoint",
            "sequence": waypoint.get("sequence", ""), "kind": waypoint.get("kind", ""),
            "latitude": waypoint.get("lat", ""), "longitude": waypoint.get("lon", ""),
            "altitude_m": waypoint.get("altitude", ""), "dwell_seconds": waypoint.get("dwell_seconds", ""),
            "cell_id": waypoint.get("cell_id", "") or "", "sensor": waypoint.get("sensor", "") or "",
            "source_frame": waypoint.get("source_frame", "") or "", "confidence": waypoint.get("confidence", "") or "",
            "rationale": waypoint.get("rationale", "") or "", "timestamp": waypoint.get("timestamp", "") or "",
            "crs": waypoint.get("crs", "EPSG:4326"),
        })
    for detection in packet.get("detections", []):
        records.append({
            "mission_id": mission_id, "record_type": "detection",
            "sequence": "", "kind": "investigate", "latitude": detection.get("lat", ""),
            "longitude": detection.get("lon", ""), "altitude_m": detection.get("altitude", ""),
            "dwell_seconds": "", "cell_id": "", "sensor": detection.get("band", ""),
            "source_frame": detection.get("source_frame", ""), "confidence": detection.get("confidence", ""),
            "rationale": detection.get("evidence", ""), "timestamp": detection.get("timestamp", "") or "",
            "crs": detection.get("crs", "EPSG:4326"),
        })
    return records


def to_geojson(packet: dict[str, Any]) -> str:
    features = []
    for record in packet_records(packet):
        properties = dict(record)
        lon, lat, altitude = float(record["longitude"]), float(record["latitude"]), float(record["altitude_m"] or 0)
        properties.pop("longitude", None)
        properties.pop("latitude", None)
        properties.pop("altitude_m", None)
        features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat, altitude]}, "properties": properties})
    import json
    return json.dumps({"type": "FeatureCollection", "features": features}, indent=2)


def to_csv(packet: dict[str, Any]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=WAYPOINT_FIELDS)
    writer.writeheader()
    writer.writerows(packet_records(packet))
    return output.getvalue()


def to_kml(packet: dict[str, Any]) -> str:
    placemarks = []
    for record in packet_records(packet):
        name = f"{record['record_type']} {record['sequence'] or record['source_frame'] or record['kind']}"
        description = " | ".join(f"{key}: {value}" for key, value in record.items() if value not in ("", None))
        placemarks.append(
            "<Placemark><name>{}</name><description>{}</description><Point><coordinates>{},{},{}</coordinates></Point></Placemark>".format(
                escape(name), escape(description), record["longitude"], record["latitude"], record["altitude_m"] or 0
            )
        )
    return "<?xml version=\"1.0\" encoding=\"UTF-8\"?><kml xmlns=\"http://www.opengis.net/kml/2.2\"><Document><name>{}</name>{}</Document></kml>".format(
        escape(str(packet.get("mission_id", "KairoDrishti field packet"))), "".join(placemarks)
    )


def to_gpx(packet: dict[str, Any]) -> str:
    points = []
    for record in packet_records(packet):
        label = f"{record['record_type']} {record['sequence'] or record['source_frame'] or record['kind']}"
        points.append(
            "<rtept lat=\"{}\" lon=\"{}\"><ele>{}</ele><name>{}</name><type>{}</type></rtept>".format(
                record["latitude"], record["longitude"], record["altitude_m"] or 0,
                escape(label), escape(str(record["kind"])),
            )
        )
    return "<?xml version=\"1.0\" encoding=\"UTF-8\"?><gpx version=\"1.1\" creator=\"KairoDrishti\" xmlns=\"http://www.topografix.com/GPX/1/1\"><metadata><name>{}</name></metadata><rte>{}</rte></gpx>".format(
        escape(str(packet.get("mission_id", "KairoDrishti field packet"))), "".join(points)
    )


def export_packet(packet: dict[str, Any], format_name: str) -> tuple[str, str, str]:
    format_name = format_name.lower().lstrip(".")
    exporters = {"geojson": (to_geojson, "application/geo+json", "geojson"),
                 "json": (to_geojson, "application/geo+json", "geojson"),
                 "csv": (to_csv, "text/csv", "csv"),
                 "kml": (to_kml, "application/vnd.google-earth.kml+xml", "kml"),
                 "gpx": (to_gpx, "application/gpx+xml", "gpx")}
    if format_name not in exporters:
        raise ValueError("unsupported export format; use geojson, csv, kml, or gpx")
    function, content_type, extension = exporters[format_name]
    return function(packet), content_type, extension
