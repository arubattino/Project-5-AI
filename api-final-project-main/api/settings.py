import os

# Run API in Debug mode
API_DEBUG = True

# We will store images uploaded by the user on this folder
UPLOAD_FOLDER = "static/uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# REDIS settings
# Queue name
REDIS_QUEUE = "app_queue"
# Port
REDIS_PORT = 6379
# DB Id
REDIS_DB_ID = 0
# Host IP
REDIS_IP = "redis"
# Sleep parameters which manages the
# interval between requests to our redis queue
API_SLEEP = 0.05

# ERROR MESSASES
ERROR_FILE_NOT_IN_REQUEST = "No file part"
ERROR_NO_VIDEO_PROVIDED = "No video selected for uploading"
ERROR_ALLOWED_VIDEO = "Allowed video types are -> mp4, webm, ogg"
ERROR_ALLOWED_URL_YOUTUBE = "Not valid YouTube link"
