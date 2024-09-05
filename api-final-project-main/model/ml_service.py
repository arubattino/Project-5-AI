from tensorflow import keras

import time
import redis
import json
import settings
import os
import numpy as np
import cv2
from middleware import process_video

# from time import time as time2

# Connect to Redis and assign to variable `db``
# Make use of settings.py module to get Redis settings like host, port, etc.
db = redis.Redis(
    host=settings.REDIS_IP, port=settings.REDIS_PORT, db=settings.REDIS_DB_ID
)

assert db.ping, "Unable to connect to redis"

model = keras.models.load_model(settings.PATH_MODEL)


def predict(directory_video, num_scene):
    """
    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.

    Parameters
    ----------
    image_name : str
        Image filename.

    Returns
    -------
    class_name, pred_probability : tuple(str, float)
        Model predicted class as a string and the corresponding confidence
        score as a number.
    """

    # return 'CS', 'Static'
    folder = os.path.join(settings.UPLOAD_FOLDER, directory_video, str(num_scene))

    frames = []
    for idx in range(settings.FRAMES):
        idx = str(idx) + ".jpg"
        img_path = os.path.join(folder, idx)
        img = cv2.imread(img_path)

        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, settings.SHAPE)
        frame = keras.applications.resnet50.preprocess_input(frame)
        frames.append(frame)

    frames = np.expand_dims(frames, axis=0)

    prediction = model.predict(np.array(frames))

    index_max_movement = prediction[0].argmax()
    class_movement = settings.MODEL_CLASSES_MOVEMENT[index_max_movement]

    index_max_scale = prediction[1].argmax()
    class_scale = settings.MODEL_CLASSES_SCALE[index_max_scale]

    return class_scale, class_movement


def classify_process():
    """
    Loop indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.

    Load video from the corresponding folder based on the video path
    received, then, run our ML model to get predictions.
    """
    while True:
        #   1. Take a new job from Redis
        #   2. Run your ML model on the given data
        #   3. Store model prediction in a dict, and each scene with the following shape:
        #      {
        #         "from": str,
        #         "to": str,
        #         "dir": str
        #         "scale": str
        #         "movement": str
        #      }
        #   4. Store the results on Redis using the original job ID as the key
        #      so the API can match the results it gets to the original job
        #      sent

        _, job_data_json = db.brpop(settings.REDIS_QUEUE)

        print("The video was received correctly by the model", flush=True)

        job_data = json.loads(job_data_json)

        video_name = job_data["video_name"]
        # the directory of the movie is the hash
        directory_video = os.path.splitext(video_name)[0]

        # video path
        video_path = os.path.join(settings.UPLOAD_FOLDER, directory_video, video_name)

        # get scenes of the movie and generate the thumbnail of each scenes
        scenes = process_video(video_path)

        # iterate throw scenes and build the info of each scene
        # we include the scale and movement predict
        info_scenes = {}
        for num_scene, scene in enumerate(scenes):
            # prectic scale and movement
            st = time.time()
            scale, movement = predict(directory_video, num_scene)
            ft = time.time()
            print(f"prediction scene {num_scene + 1 } took {ft-st} seconds", flush=True)
            info_scene = {
                "from": scene["from"],
                "to": scene["to"],
                "dir": os.path.join(directory_video, str(num_scene)),
                "scale": scale,
                "movement": movement,
                "frames": settings.FRAMES,
            }

            info_scenes[num_scene] = info_scene

        db.set(job_data["id"], json.dumps(info_scenes))

        # sleep for a bit at the end
        time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    # Now launch process
    print("Launching ML service...")
    classify_process()
