import numpy as np
import cv2


# To Convert the Transcription Data to Document (index, points, transcription) Format
def read_boxes_and_tr_data(tr_image):
    res = []
    for i in range(0, len(tr_image["text"])):
        # Clockwise 8 points
        x1 = tr_image["left"][i]
        y1 = tr_image["top"][i]

        x2 = x1 + tr_image["width"][i]
        y2 = y1

        x3 = x2
        y3 = y1 + tr_image["height"][i]

        x4 = x1
        y4 = y3

        points = [float(x1), float(y1), float(x2), float(y2), float(x3), float(y3), float(x4), float(y4)]
        transcription = tr_image["text"][i]

        res.append((i, points, transcription))

    return res


# To Sort Document Format
def sort_boxes_and_tr_data(data, left_right_first=False):
    def compare_key(x):
        #  x is (index, points, transcription)
        points = x[1]
        box = np.array([[points[0], points[1]], [points[2], points[3]], [points[4], points[5]], [points[6], points[7]]],
                       dtype=np.float32)
        rect = cv2.minAreaRect(box)
        center = rect[0]
        if left_right_first:
            return center[0], center[1]
        else:
            return center[1], center[0]

    data = sorted(data, key=compare_key)
    return data


# To Whitespace Empty Transcriptions
def whitespace_empty_trs(boxes_and_transcripts_data):
    nboxes = []
    for i in range(len(boxes_and_transcripts_data)):
        index = boxes_and_transcripts_data[i][0]
        points = boxes_and_transcripts_data[i][1]

        if len(boxes_and_transcripts_data[i][2]) == 0:
            transcript = ' '
        else:
            transcript = boxes_and_transcripts_data[i][2]

        nboxes.append((index, points, transcript))

    return nboxes


# To Obtain Boxes and Transcripts Separately
def split_boxes_and_transcripts(boxes_and_transcripts_data):
    boxes, transcripts = [], [],
    for index, points, transcript in boxes_and_transcripts_data:
        boxes.append(points)
        transcripts.append(transcript)

    return boxes, transcripts


# To Calculate Relation Features
def relation_features_between_ij_nodes(boxes_num, i, min_area_boxes, relation_features, transcript_i, transcripts):
    for j in range(boxes_num):
        transcript_j = transcripts[j]

        rect_output_i = min_area_boxes[i]
        rect_output_j = min_area_boxes[j]

        center_i = rect_output_i[0]
        center_j = rect_output_j[0]

        width_i, height_i = rect_output_i[1]
        width_j, height_j = rect_output_j[1]

        relation_features[i, j, 0] = np.abs(center_i[0] - center_j[0]) if np.abs(
            center_i[0] - center_j[0]) is not None else -1  # x_ij

        relation_features[i, j, 1] = np.abs(center_i[1] - center_j[1]) if np.abs(
            center_i[1] - center_j[1]) is not None else -1  # y_ij

        relation_features[i, j, 2] = width_i / (height_i) if width_i / (height_i) is not None else -1  # w_i/h_i

        relation_features[i, j, 3] = height_j / (height_i) if height_j / (height_i) is not None else -1  # h_j/h_i
        relation_features[i, j, 4] = width_j / (height_i) if width_j / (height_i) is not None else -1  # w_j/h_i
        relation_features[i, j, 5] = len(transcript_j) / (len(transcript_i)) if len(transcript_j) / (
            len(transcript_i)) is not None else -1  # T_j/T_i


# To Normalize Relation Features
def normalize_relation_features(feat: np.ndarray, width: int, height: int):
    np.clip(feat, 1e-8, np.inf)
    feat[:, :, 0] = feat[:, :, 0] / width
    feat[:, :, 1] = feat[:, :, 1] / height

    for i in range(2, 6):
        feat_ij = feat[:, :, i]
        max_value = np.max(feat_ij)
        min_value = np.min(feat_ij)
        if max_value != min_value:
            feat[:, :, i] = feat[:, :, i] - min_value / (max_value - min_value)
    return feat
