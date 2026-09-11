from app.server import scenario


def test_dashboard_scenario_preserves_waypoint_altitude_and_coordinates():
    payload = scenario()
    assert payload["coordinate_system"].startswith("WGS84")
    assert payload["waypoints"][-1]["kind"] == "return_home"
    assert all("lat" in waypoint and "lon" in waypoint and "altitude" in waypoint
               for waypoint in payload["waypoints"])


def test_dashboard_scenario_contains_operational_layers():
    payload = scenario()
    assert {cell["id"] for cell in payload["cells"]} >= {"RIDGE-01", "STREAM-02"}
    assert payload["no_fly"][0]["height"] > 0
    assert payload["truth"]["confidence"] == .84
