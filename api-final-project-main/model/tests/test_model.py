import unittest

import ml_service

import os


class TestMLService(unittest.TestCase):
    def test_predict(self):
        ml_service.settings.UPLOAD_FOLDER = "tests"
        directory = "test_data/e3e0aafc5c1dab9f7743e63dcf4c8a83"
        num_scene = 0
        class_scale, class_movement = ml_service.predict(directory, num_scene)
        self.assertEqual(class_scale, "LS")
        self.assertEqual(class_movement, "Motion")


if __name__ == "__main__":
    unittest.main()
