import time
import redis
import settings
import uuid
import json

# TODO
# Connect to Redis and assign to variable `db``
# Make use of settings.py module to get Redis settings like host, port, etc.
db = redis.Redis(
    host=settings.REDIS_IP, port=settings.REDIS_PORT, db=settings.REDIS_DB_ID
)

assert db.ping, "Unable to connect to redis"


def model_predict(video_name):
    """
    Receives an video name and queues the job into Redis.
    Will loop until getting the answer from our ML service.

    Parameters
    ----------
    video_name : str
        video name.

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
    #    "video_name": str,
    # }

    job_data = {"id": job_id, "video_name": video_name}

    # Send the job to the model service using Redis

    db.rpush(settings.REDIS_QUEUE, json.dumps(job_data))
    print("The video was sent to the model by redis", flush=True)
    # Loop until we received the response from our ML model
    while True:
        # Attempt to get model  using job_id

        if db.exists(job_id):
            output = db.get(job_id)
            info_scenes = json.loads(output)
            db.delete(job_id)
            break

        # Sleep some time waiting for model results
        time.sleep(settings.API_SLEEP)

    return info_scenes
