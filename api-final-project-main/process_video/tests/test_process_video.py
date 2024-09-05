import os
from unittest import TestCase
import settings
from scenes_process import ScenesProcess


class TestIntegration(TestCase):
    def test_process_video(self):
        video_path = "tests/deab000c5012b3e9dafe8262046a556c/deab000c5012b3e9dafe8262046a556c.mp4"
        frames_by_scene = 8
        process_video = ScenesProcess(video_path, frames_by_scene=frames_by_scene)
        scenes = process_video.process()

        dir_scenes = len(
            os.listdir(
                os.path.join(settings.UPLOAD_FOLDER, "deab000c5012b3e9dafe8262046a556c")
            )
        )

        images_in_scenes = len(
            os.listdir(
                os.path.join(
                    settings.UPLOAD_FOLDER, "deab000c5012b3e9dafe8262046a556c", "0"
                )
            )
        )

        self.assertEqual(dir_scenes, 5)
        self.assertEqual(dir_scenes, len(scenes))
        self.assertEqual(images_in_scenes, frames_by_scene)
