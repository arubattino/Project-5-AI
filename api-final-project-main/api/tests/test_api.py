import json
from unittest import TestCase
from unittest.mock import patch
import settings
from app import app


class TestIntegration(TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_predict_bad_parameters(self):
        response = self.client.post(
            "/predict",
            data=json.dumps({"not_a_file": "blabla"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.get_data(as_text=True))
        self.assertEqual(data["error"], settings.ERROR_FILE_NOT_IN_REQUEST)


class TestEnpointsAvailability(TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_index(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        response = self.client.post("/")
        self.assertEqual(response.status_code, 405)

    def test_predict(self):
        response = self.client.get("/predict")
        # Method not allowed
        self.assertEqual(response.status_code, 405)
        response = self.client.post("/predict")
        # Bad args
        self.assertEqual(response.status_code, 400)
