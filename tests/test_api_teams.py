from __future__ import annotations

from fastapi.testclient import TestClient

from db.session import get_db
from main import app


def test_teams_endpoint_returns_sorted_payload():
    class FakeScalarResult:
        def __init__(self, rows):
            self._rows = rows

        def all(self):
            return self._rows

    class Team:
        def __init__(self, id_, name, hockeytech_id, city, conference, division, logo_url, active):
            self.id = id_
            self.name = name
            self.hockeytech_id = hockeytech_id
            self.city = city
            self.conference = conference
            self.division = division
            self.logo_url = logo_url
            self.active = active

    class FakeSession:
        def scalars(self, _stmt):
                return FakeScalarResult(
                        [
                            Team(1, "A Team", 201, "A City", "West", "A", None, True),
                            Team(2, "B Team", 202, "B City", "East", "B", None, True),
                        ]
                    )

    def fake_get_db():
        yield FakeSession()

    app.dependency_overrides[get_db] = fake_get_db
    client = TestClient(app)
    try:
        response = client.get("/teams")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200

    body = response.json()
    assert isinstance(body, list)
    first = body[0]
    assert {"id", "name", "hockeytech_id", "city", "conference", "division", "logo_url", "active"} <= set(first.keys())

    names = [row["name"] for row in body]
    assert names == sorted(names)
