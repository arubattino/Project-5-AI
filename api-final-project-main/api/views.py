import utils
import settings
import os
import time

from flask import Blueprint, flash, redirect, render_template, request, url_for, jsonify

router = Blueprint("app_router", __name__, template_folder="templates")


@router.route("/", methods=["GET"])
def index():
    """
    Index endpoint, renders our HTML code.
    """
    return render_template("index.html")


@router.route("/predict_youtube", methods=["POST"])
def predict_youtube():
    """
    Endpoint used to get predictions from youtube url.

    Parameters
    ----------
    url: str
        youtube url we want to get predictions from.

    Returns
    -------
    flask.Response
        JSON response from our API having the following format:
            {
                "success": bool,
                "scenes": json,
                "file_name": str,
                "dir": str
            }

        - "success" will be True if the input file is valid and we get a
          prediction from our ML model.
        - "scenes" info for each scene.
                {
                    'from': str,
                    'to': str,
                    'dir': str,
                    'scale': str,
                    'movement': str
                }
        - "file_name" the video name
        - "dir" the directory where the video is saved
    """
    st = time.time()
    # No url received
    if "url" not in request.form:
        return jsonify({"error": settings.ERROR_FILE_NOT_IN_REQUEST}), 400

    youtube_url = request.form.get("url")

    if not (utils.allowed_url(youtube_url)):
        return jsonify({"error": settings.ERROR_ALLOWED_URL_YOUTUBE}), 400

    try:
        directory, video_name = utils.process_youtube_url(youtube_url)
    except BaseException as error:
        return jsonify({"error": str(error)}), 400

    # get scene prediction
    no_cache = request.form.get("no_cache", False)
    rpse = utils.check_json(directory, video_name, no_cache)

    ft = time.time()
    print(f"all prediction took {ft-st} seconds", flush=True)
    return jsonify(rpse)


@router.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint used to get predictions.

    Parameters
    ----------
    file : str
        Input image we want to get predictions from.

    Returns
    -------
    flask.Response
        JSON response from our API having the following format:
            {
                "success": bool,
                "scenes": json,
                "file_name": str,
                "dir": str
            }

        - "success" will be True if the input file is valid and we get a
          prediction from our ML model.
        - "scenes" info for each scene.
                {
                    'from': str,
                    'to': str,
                    'dir': str,
                    'scale': str,
                    'movement': str
                }
        - "file_name" the video name
        - "dir" the directory where the video is saved
    """
    st = time.time()
    # No file received
    if "file" not in request.files:
        return jsonify({"error": settings.ERROR_FILE_NOT_IN_REQUEST}), 400

    # File received but no filename is provided
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": settings.ERROR_NO_VIDEO_PROVIDED}), 400

    if not (file and utils.allowed_file(file.filename)):
        return jsonify({"error": settings.ERROR_ALLOWED_VIDEO}), 400

    directory, video_name = utils.directory_file_hash(file)

    # create directory for the file
    os.makedirs(os.path.join(settings.UPLOAD_FOLDER, directory), exist_ok=True)

    # if file not exist save it
    if not os.path.exists(os.path.join(settings.UPLOAD_FOLDER, directory, video_name)):
        file.save(os.path.join(settings.UPLOAD_FOLDER, directory, video_name))

    # get scene prediction
    no_cache = request.form.get("no_cache", False)
    rpse = utils.check_json(directory, video_name, no_cache)

    ft = time.time()
    print(f"all prediction took {ft-st} seconds", flush=True)

    return jsonify(rpse)
