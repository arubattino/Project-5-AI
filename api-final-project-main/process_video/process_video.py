import time
import redis
import json
import settings
from scenes_process import ScenesProcess

# Connect to Redis and assign to variable `db``
# Make use of settings.py module to get Redis settings like host, port, etc.
db = redis.Redis(
    host=settings.REDIS_IP, port=settings.REDIS_PORT, db=settings.REDIS_DB_ID
)

assert db.ping, "Unable to connect to redis"


def process_video():
    while True:

        _, job_data_json = db.brpop(settings.REDIS_VIDEO_QUEUE)

        print("The video is ready to be process by video scene", flush=True)
        job_data = json.loads(job_data_json)

        video_path = job_data["video_path"]
        frames_by_scene = job_data["frames"]

        secenes_process = ScenesProcess(video_path, frames_by_scene=frames_by_scene)
        scenes = secenes_process.process()

        db.set(job_data["id"], json.dumps(scenes))

        # sleep for a bit at the end
        time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    # Now launch process
    print("Launching process video service...")
    process_video()
