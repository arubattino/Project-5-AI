import os
from unittest import TestCase
from unittest.mock import patch

from werkzeug.datastructures import FileStorage

import utils
from app import app


class TestUtils(TestCase):
    def test_allowed_file(self):
        self.assertFalse(utils.allowed_file("cat.JPG"))
        self.assertFalse(utils.allowed_file("cat.jpeg"))
        self.assertFalse(utils.allowed_file("cat.JPEG"))
        self.assertFalse(utils.allowed_file("../../car.PNG"))
        self.assertFalse(utils.allowed_file("/usr/var/src/car.gif"))

        self.assertTrue(utils.allowed_file("cat.mp4"))
        self.assertTrue(utils.allowed_file("cat.Mp4"))
        self.assertTrue(utils.allowed_file("cat.ogg"))
        self.assertTrue(utils.allowed_file("cat.webm"))

    def test_get_file_hash(self):
        filename = "tests/movie.mp4"
        md5_filename = "deab000c5012b3e9dafe8262046a556c.mp4"
        with open(filename, "rb") as fp:
            file = FileStorage(fp)

            # Check the new filename is correct
            _, new_filename = utils.directory_file_hash(file)
            self.assertEqual(md5_filename, new_filename, new_filename)

            # Check the file content is still readable!
            self.assertTrue(file.read() != b"")

    def test_valid_url_youtube(self):
        url = "https://www.youtube.com/watch?v=qXYb8R3_B0k"

        self.assertTrue(utils.allowed_url(url))
        self.assertFalse(utils.allowed_url("blabla"))

    @patch("settings.UPLOAD_FOLDER")
    def test_process_youtube_url(self, mock):
        directory_valid = "01052cd8a3965f0a01fa8caaf91c5c72"
        filename_valid = "01052cd8a3965f0a01fa8caaf91c5c72.mp4"
        mock.return_value = "tmp/"
        url_youtube = "https://www.youtube.com/watch?v=z2T-Rh838GA"
        directory, filename = utils.process_youtube_url(url_youtube)

        self.assertEqual(filename_valid, filename)
        self.assertEqual(directory_valid, directory)
        self.assertTrue(os.path.exists(os.path.join(mock, directory, filename_valid)))
