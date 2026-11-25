from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["neighborhood_count"] > 0


def test_baseline_endpoint():
    resp = client.get("/baseline")
    assert resp.status_code == 200
    neighborhoods = resp.json()["neighborhoods"]
    assert len(neighborhoods) > 0
    assert "NAME" in neighborhoods[0]
    assert "baseline_pred_c" in neighborhoods[0]


def test_mitigation_cools_and_spillover_propagates():
    resp = client.post(
        "/predict",
        json={
            "neighborhood_name": "Midtown",
            "trees_added": 150,
            "park_added_acres": 4.0,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["temp_change_c"] < 0.0
    assert data["avg_neighbor_spillover_c"] < 0.0


def test_new_development_warms_neighborhood():
    resp = client.post(
        "/predict",
        json={
            "neighborhood_name": "Midtown",
            "new_dev_acres": 5.0,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["temp_change_c"] > 0.0


def test_invalid_neighborhood_returns_404():
    resp = client.post("/predict", json={"neighborhood_name": "Atlantis"})
    assert resp.status_code == 404