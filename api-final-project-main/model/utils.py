import yaml


def validate_config(config):
    """
    checks the model settings

    Parameters
    ----------
    config : dict
        the model settings
    """
    if "path_model" not in config:
        raise ValueError("Missing weights")

    if "classes_movement" not in config:
        raise ValueError("Missing the movement classes")

    if "classes_scale" not in config:
        raise ValueError("Missing the scale classes")

    if "shape" not in config:
        raise ValueError("Missing the shape")

    if "frames" not in config:
        raise ValueError("Missing the frames")


def load_config(config_file_path):
    """
    Loads the config model

    Parameters
    ----------
    config_file_path : str

    Returns
    -------
    config : dict
        model settings as a Python dict.
    """

    with open(config_file_path, "r") as stream:
        config = yaml.safe_load(stream)

    validate_config(config)

    return config
