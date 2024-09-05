import os
import utils

# We will store images uploaded by the user on this folder
UPLOAD_FOLDER = "uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# REDIS
# Queue name
REDIS_QUEUE = "app_queue"
REDIS_VIDEO_QUEUE = "video_queue"
# Port
REDIS_PORT = 6379
# DB Id
REDIS_DB_ID = 0
# Host IP
REDIS_IP = "redis"
# Sleep parameters which manages the
# interval between requests to our redis queue
SERVER_SLEEP = 0.05

models_settings = utils.load_config("config_model.yml")

SHAPE = models_settings["shape"]
MODEL_CLASSES_MOVEMENT = models_settings["classes_movement"]
MODEL_CLASSES_SCALE = models_settings["classes_scale"]
PATH_MODEL = models_settings["path_model"]
FRAMES = models_settings["frames"]
