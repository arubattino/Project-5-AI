import settings, cv2, os
import numpy as np

# Standard PySceneDetect imports:
from scenedetect import VideoManager
from scenedetect import SceneManager

# For content-aware scene detection:
from scenedetect.detectors import ContentDetector
from time import time


class ScenesProcess:
    """
    Detect the video scenes, and return a list of tuples containing the
    (start, end) timecodes of each detected scene.

    Also generate directory for each scene and images for each scene
    """

    def __init__(self, video_path, frames_by_scene=8):
        self.video_path = video_path
        self.frames_by_scene = frames_by_scene
        self.scenes = []

    def process(self, threshold=30.0):
        """
        Extract scenes from the video

        Returns
        -------
        list
            list of scene with dict with this structure
            {
                to: str
                from: str
            }
        """
        st = time()
        # Create our video & scene managers, then add the detector.
        video_manager = VideoManager([self.video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))

        # Base timestamp at frame 0 (required to obtain the scene list).
        base_timecode = video_manager.get_base_timecode()

        # Improve processing speed by downscaling before processing.
        video_manager.set_downscale_factor()

        # Start the video manager and perform the scene detection.
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)

        # Each returned scene is a tuple of the (start, end) timecode.
        self.scenes = scene_manager.get_scene_list(base_timecode, start_in_scene=True)

        directory_path = os.path.dirname(self.video_path)

        # check if the scenes images are alredy created
        if len(os.listdir(directory_path)) <= len(self.scenes):
            self.__generate_thumbails_scenes()

        list_scenes = []

        for scene in self.scenes:
            info_scene = {
                "from": str(scene[0]),
                "to": str(scene[1]),
            }

            list_scenes.append(info_scene)

        ft = time()

        print(f"video scene split took {ft-st} seconds", flush=True)
        return list_scenes

    def __generate_thumbails_scenes(self):
        """
        Generate directory for each scene and images for each scene
        """
        # get the movie directory
        directory_movie = os.path.basename(os.path.dirname(self.video_path))
        num_frames_scene = {}
        for num_scene, scene in enumerate(self.scenes):

            from_frame = scene[0].get_frames()
            to_frame = scene[1].get_frames() - 1

            # choose the equidistant number of frames
            frames_to_choose = np.linspace(
                from_frame, to_frame, self.frames_by_scene, dtype=int
            )

            # we create dictionary with key with frames number and
            # value with scene number
            for num_frame in frames_to_choose:
                num_frames_scene[num_frame] = num_scene

        scene_frame = self.__get_frames(num_frames_scene)

        # create directory scene and images for each scene
        for scene, frames in scene_frame.items():
            # for scenes with less than self.frames_by_scene
            len_frames = len(frames)
            if len_frames < self.frames_by_scene:
                for i in range(len_frames, self.frames_by_scene):
                    frames.append(frames[-1])

            os.makedirs(
                os.path.join(settings.UPLOAD_FOLDER, directory_movie, str(scene)),
                exist_ok=True,
            )
            for i, frame in enumerate(frames):
                image = str(i) + ".jpg"
                if not os.path.exists(
                    os.path.join(
                        settings.UPLOAD_FOLDER, directory_movie, str(scene), image
                    )
                ):
                    cv2.imwrite(
                        os.path.join(
                            settings.UPLOAD_FOLDER, directory_movie, str(scene), image
                        ),
                        frame,
                    )

    def __get_frames(self, num_frames_scene):
        """
        Generate a dictionary with key with scene
        and value with a frames list

        Parameters
        ----------
        num_frames_scene : dict
                    dictionary with key with frames number and
                    value with scene number
        Returns
        -------
        dict
           dictionary with key with scene and value with a frames list
        """
        cap = cv2.VideoCapture(self.video_path)
        scene_frame_dic = {}
        i = 0
        while True:
            ret, frame = cap.read()
            if ret == False:
                break
            if i in num_frames_scene:
                scene = num_frames_scene[i]
                if scene in scene_frame_dic:
                    scene_frame_dic[scene].append(frame)
                else:
                    scene_frame_dic[scene] = [frame]
            i += 1
        cap.release()
        return scene_frame_dic
