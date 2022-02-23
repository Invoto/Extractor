import os
import consts.env as consts_env
import requests


def is_dev():
    return os.getenv(consts_env.KEY_IS_DEV) is not None


# Protected function to retrieve public IP from ipinfo.io
def _get_public_ip():
    endpoint = "https://ipinfo.io/json"
    response = requests.get(endpoint, verify=True)

    if response.status_code != 200:
        return None

    data = response.json()
    return data["ip"]


def get_host():
    return "0.0.0.0"


def get_port():
    return int(os.getenv(consts_env.KEY_PORT)) or 8080


def get_parse_app_id():
    return os.getenv(consts_env.KEY_PARSE_APP_ID)


def get_parse_master_key():
    return os.getenv(consts_env.KEY_PARSE_MASTER_KEY)


def get_parse_api_key():
    return os.getenv(consts_env.KEY_PARSE_API_KEY)


def get_model_chkpt_path():
    return os.getenv(consts_env.KEY_MODEL_CHKPT_PATH)
