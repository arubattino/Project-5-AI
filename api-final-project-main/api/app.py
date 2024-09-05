import settings
from flask import Flask
from views import router

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = settings.UPLOAD_FOLDER
app.secret_key = "secret key"
app.register_blueprint(router)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1000 * 1000

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=settings.API_DEBUG)
