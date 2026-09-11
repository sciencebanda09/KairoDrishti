"""Optional, cached OpenSky airspace awareness client.

OpenSky is advisory context only. It is never treated as an authoritative
UAS geofence and never blocks offline mission planning.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class OpenSkyCache:
    def __init__(self, *, enabled: bool = False, cache_path: str | Path = "runs/cache/opensky.json",
                 ttl_seconds: int = 30, timeout_seconds: float = 3.0) -> None:
        self.enabled = enabled
        self.cache_path = Path(cache_path)
        self.ttl_seconds = ttl_seconds
        self.timeout_seconds = timeout_seconds
        self.last: dict[str, Any] | None = None

    def _read_cache(self) -> dict[str, Any] | None:
        if self.last is not None:
            return self.last
        if not self.cache_path.exists():
            return None
        try:
            self.last = json.loads(self.cache_path.read_text(encoding="utf-8"))
            return self.last
        except (OSError, json.JSONDecodeError):
            return None

    def _write_cache(self, payload: dict[str, Any]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.last = payload

    @staticmethod
    def _aircraft(states: list[list[Any]] | None) -> list[dict[str, Any]]:
        result = []
        for state in states or []:
            if len(state) < 11 or state[5] is None or state[6] is None:
                continue
            result.append({
                "icao24": state[0], "callsign": (state[1] or "").strip(),
                "longitude": state[5], "latitude": state[6],
                "altitude_m": state[7] if state[7] is not None else state[13] if len(state) > 13 else None,
                "velocity_mps": state[9], "heading_deg": state[10],
                "vertical_rate_mps": state[11] if len(state) > 11 else None,
                "last_contact": state[3],
            })
        return result

    def snapshot(self, bounds: dict[str, float] | None = None, *, force: bool = False) -> dict[str, Any]:
        cached = self._read_cache()
        now = time.time()
        if not self.enabled:
            return {"source": "disabled", "available": False, "stale": True, "aircraft": [],
                    "message": "OpenSky disabled; configured no-fly volumes remain authoritative."}
        if cached and not force and now - float(cached.get("fetched_at", 0)) < self.ttl_seconds:
            return {**cached, "source": "cache", "stale": False}
        params = {}
        if bounds:
            params = {key: bounds[key] for key in ("lamin", "lomin", "lamax", "lomax") if key in bounds}
        url = "https://opensky-network.org/api/states/all"
        if params:
            url = f"{url}?{urlencode(params)}"
        try:
            request = Request(url, headers={"User-Agent": "KairoDrishti/1.0 SAR research client"})
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
            result = {"source": "opensky", "available": True, "stale": False,
                      "fetched_at": now, "time": payload.get("time"),
                      "aircraft": self._aircraft(payload.get("states")),
                      "message": "OpenSky ADS-B context; not an authoritative UAS geofence."}
            self._write_cache(result)
            return result
        except Exception as error:  # network is optional and must never break offline work
            if cached:
                return {**cached, "source": "cache", "available": True, "stale": True,
                        "message": f"OpenSky unavailable; using cached snapshot ({error})."}
            return {"source": "unavailable", "available": False, "stale": True, "aircraft": [],
                    "message": f"OpenSky unavailable; offline no-fly configuration remains active ({error})."}
