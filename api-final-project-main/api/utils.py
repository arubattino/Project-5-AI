import os, io, re, requests
import hashlib
import settings
from middleware import model_predict
from pytube import YouTube
import json
import time


def allowed_file(filename):
    """
    Checks if the format for the file received is acceptable. For this
    particular case, we must accept only image files.

    Parameters
    ----------
    filename : str
        Filename from werkzeug.datastructures.FileStorage file.

    Returns
    -------
    bool
        True if the file is an image, False otherwise.
    """

    split_file = os.path.splitext(filename)
    return str(split_file[1]).lower() in [".mp4", ".webm", ".ogg"]


def allowed_url(url):
    """
    Checks if the url youtube format is valid.

    Parameters
    ----------
    url : str
        the youtube url.

    Returns
    -------
    bool
        True if the url is valid, False otherwise.
    """
    youtube_regex = r"^((?:https?:)?\/\/)?((?:www|m)\.)?((?:youtube\.com|youtu.be))(\/(?:[\w\-]+\?v=|embed\/|v\/)?)([\w\-]+)(\S+)?$"

    youtube_regex_match = re.match(youtube_regex, url)

    if youtube_regex_match:
        return True

    return False


def directory_file_hash(file, buffer=False):
    """
    Returns a new filename based on the file content using MD5 hashing.
    It uses hashlib.md5() function from Python standard library to get
    the hash.

    Parameters
    ----------
    file : werkzeug.datastructures.FileStorage
        File sent by user.

    Returns
    -------
        A tuple having the directory of the file and the file name.
    """

    if not buffer:
        split_file = os.path.splitext(file.filename)
        file_ext = str(split_file[1])
    else:
        file_ext = ".mp4"

    data = file.read()
    md5hash = hashlib.md5(data).hexdigest()
    file.seek(0)

    return str(md5hash), str(md5hash) + file_ext


def process_youtube_url(url):
    """
    It process the youtube url video with the library pytube, it saves the video as mp4
    Returns a new filename based on the file content using MD5 hashing.
    It uses hashlib.md5() function from Python standard library to get
    the hash.

    Parameters
    ----------
    file : werkzeug.datastructures.FileStorage
        File sent by user.

    Returns
    -------
        A tuple having the directory of the file and the file name.
    """
    st = time.time()

    buf = io.BytesIO()

    yt = YouTube(url)

    # we get the mp4 from youtube
    stream = yt.streams.filter(
        progressive=True, file_extension="mp4"
    ).get_lowest_resolution()

    # we save the file in memory
    stream.stream_to_buffer(buf)
    buf.seek(0)
    # read the file
    # data = buf.read()
    # get the md5
    directory, filename = directory_file_hash(buf, buffer=True)

    # create directory for the file
    os.makedirs(os.path.join(settings.UPLOAD_FOLDER, directory), exist_ok=True)

    if not os.path.exists(os.path.join(settings.UPLOAD_FOLDER, directory, filename)):
        # save the file in the directory
        stream.download(
            output_path=os.path.join(settings.UPLOAD_FOLDER, directory),
            filename=filename,
        )

    ft = time.time()

    print(f"pytube took {ft-st} seconds", flush=True)

    return directory, filename


def get_prediction(video_name):

    # the directory of the movie is the hash
    directory_movie = os.path.splitext(video_name)[0]

    # get the video path
    video_path = os.path.join(settings.UPLOAD_FOLDER, directory_movie, video_name)

    assert os.path.exists(os.path.join(video_path)), "The file doesn't exists"

    # we call model_predict from middleware to predict each scene
    info_scenes = model_predict(video_name)

    return info_scenes


def download_json(file):
    filename = "data.json"
    path = os.path.join(settings.UPLOAD_FOLDER, file["dir"], filename)
    with open(path, "w") as fp:
        json.dump(file, fp)


def check_json(directory, video_name, no_cache):

    print("The video was recived correctly in the API", flush=True)

    path = os.path.join(settings.UPLOAD_FOLDER, directory, "data.json")

    if not os.path.exists(path) or no_cache == "true":
        scenes_predictions = get_prediction(video_name)
        rpse = {
            "success": True,
            "scenes": scenes_predictions,
            "file_name": video_name,
            "dir": directory,
        }

        download_json(rpse)

    else:
        with open(path, "rb") as f:
            json_file = f.read()
            rpse = json.loads(json_file)

    return rpse
