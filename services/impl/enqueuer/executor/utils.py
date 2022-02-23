import cv2
import requests
from . import config


def read_image(image_file_path):
    return cv2.imread(image_file_path)


def transcribe_image(image_file_path: str):
    r_headers = {
        "apikey": config.OCR_API_API_KEY,
    }

    r_payload = {
        "language": "eng",
        "isOverlayRequired": "true",
    }

    r_files = {
        "file": open(image_file_path, "rb")
    }

    response = requests.post(config.OCR_API_URL_ENDPOINT, headers=r_headers, data=r_payload, files=r_files)
    lines = response.json()["ParsedResults"][0]["TextOverlay"]["Lines"]

    # Converting to the Format Required.
    tr_image = {
        "text": [],
        "top": [],
        "left": [],
        "width": [],
        "height": [],
    }

    for line in lines:
        tr_image["text"].append(line["LineText"])

        xmins = []
        ymins = []
        xmaxs = []
        ymaxs = []
        for word in line["Words"]:
            xmins.append(float(word["Left"]))
            ymins.append(float(word["Top"]))
            xmaxs.append(float(word["Left"]) + float(word["Width"]))
            ymaxs.append(float(word["Top"]) + float(word["Height"]))

        xmin = min(xmins)
        xmax = max(xmaxs)
        ymin = min(ymins)
        ymax = max(ymaxs)

        width = xmax - xmin
        height = ymax - ymin

        tr_image["left"].append(xmin)
        tr_image["top"].append(ymin)
        tr_image["width"].append(width)
        tr_image["height"].append(height)

    return tr_image
