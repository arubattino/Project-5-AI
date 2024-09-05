import time
import redis
import settings
import uuid
import json

# Connect to Redis and assign to variable `db``
# Make use of settings.py module to get Redis settings like host, port, etc.
db = redis.Redis(
    host=settings.REDIS_IP, port=settings.REDIS_PORT, db=settings.REDIS_DB_ID
)

assert db.ping, "Unable to connect to redis"


def process_video(video_path):
    """
    Receives an video name and queues the job into Redis.
    Will loop until getting the answer from our process video service.

    Parameters
    ----------
    video_path : str
        video_path.

    Returns
    -------
    prediction, score : tuple(str, float)
        Model predicted class as a string and the corresponding confidence
        score as a number.
    """
    # Assign an unique ID for this job and add it to the queue.
    # We need to assing this ID because we must be able to keep track
    # of this particular job across all the services

    job_id = str(uuid.uuid4())

    # Create a dict with the job data we will send through Redis having the
    # following shape:
    # {
    #    "id": str,
    #    "video_path": str,
    # }

    job_data = {"id": job_id, "video_path": video_path, "frames": settings.FRAMES}

    # Send the job to the process video service using Redis

    db.rpush(settings.REDIS_VIDEO_QUEUE, json.dumps(job_data))

    # Loop until we received the response from our ML model
    while True:
        # Attempt to get model  using job_id

        if db.exists(job_id):
            output = db.get(job_id)
            info_scenes = json.loads(output)
            db.delete(job_id)
            break

        # Sleep some time waiting for model results
        time.sleep(settings.SERVER_SLEEP)

    return info_scenes
