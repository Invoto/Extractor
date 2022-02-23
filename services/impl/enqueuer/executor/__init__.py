
def predict_image(job_id: str, image_file_path: str, model_chkpt_path: str, parse_app_id: str, parse_api_key: str, parse_master_key: str):
    # Imports
    from parse_rest.connection import register
    from . import utils
    from . import helpers
    from . import config
    import numpy as np
    import cv2
    from torchtext.data import Field, RawField
    from .pick_pytorch.utils.class_utils import keys_vocab_cls
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from .pick_pytorch.model import pick as pick_arch_module
    from .pick_pytorch.utils.util import iob_index_to_str, text_index_to_str
    from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans
    from dao.wrappers.jobs import JobHandlerWrapper
    from dao.models.jobs import JobStatus

    # Creating the Job Manager.
    job_wrapper: JobHandlerWrapper = JobHandlerWrapper(job_id)

    try:
        # Registers Parse Service
        register(parse_app_id, parse_api_key, master_key=parse_master_key)

        # Updating Job Status.
        job_wrapper.update_job_status(JobStatus.ONGOING)
        job_wrapper.progress_update("Execution Started.")

        job_wrapper.progress_update("Reading and Transcribing Invoice.")
        # Opening the Image.
        image = utils.read_image(image_file_path)
        # Performing OCR on image.
        tr_image = utils.transcribe_image(image_file_path)
        job_wrapper.progress_update("SUCCESS: Reading and Transcribing Invoice.")

        job_wrapper.progress_update("Preprocessing Transcriptions.")
        # Formatting and Sorting
        boxes_and_transcripts_data = helpers.read_boxes_and_tr_data(tr_image)
        boxes_and_transcripts_data = helpers.sort_boxes_and_tr_data(boxes_and_transcripts_data)

        # Splitting Up Transcription Data and Defining Features
        boxes_and_transcripts_data = helpers.whitespace_empty_trs(boxes_and_transcripts_data)
        boxes, transcripts = helpers.split_boxes_and_transcripts(boxes_and_transcripts_data)
        job_wrapper.progress_update("SUCCESS: Preprocessing Transcriptions.")

        job_wrapper.progress_update("Preprocessing Image and Metadata.")
        boxes_num = min(len(boxes), config.MAX_BOXES_NUM)
        transcript_len = min(max([len(t) for t in transcripts[:boxes_num]]), config.MAX_TRANSCRIPT_LEN)
        mask = np.zeros((boxes_num, transcript_len), dtype=int)

        relation_features = np.zeros((boxes_num, boxes_num, 6))

        height, width, _ = image.shape

        image = cv2.resize(image, config.RESIZE_IMAGE_DIMS, interpolation=cv2.INTER_LINEAR)
        x_scale = config.RESIZE_IMAGE_DIMS[0] / width
        y_scale = config.RESIZE_IMAGE_DIMS[1] / height
        job_wrapper.progress_update("SUCCESS: Preprocessing Image and Metadata.")

        job_wrapper.progress_update("Calculating Features.")
        # get min area box for each boxes, for calculate initial relation features
        min_area_boxes = [cv2.minAreaRect(np.array(box, dtype=np.float32).reshape(4, 2)) for box in boxes[:boxes_num]]

        # Resizing and Calculating Initial Relation Features
        resized_boxes = []
        for i in range(boxes_num):
            box_i = boxes[i]
            transcript_i = transcripts[i]

            # get resized images's boxes coordinate, used to ROIAlign in Encoder layer
            resized_box_i = [int(np.round(pos * x_scale)) if i % 2 == 0 else int(np.round(pos * y_scale)) for i, pos in
                             enumerate(box_i)]

            resized_rect_output_i = cv2.minAreaRect(np.array(resized_box_i, dtype=np.float32).reshape(4, 2))
            resized_box_i = cv2.boxPoints(resized_rect_output_i)
            resized_box_i = resized_box_i.reshape((8,))
            resized_boxes.append(resized_box_i)

            # enumerate each box, calculate relation features between i and other nodes.
            helpers.relation_features_between_ij_nodes(boxes_num, i, min_area_boxes, relation_features, transcript_i, transcripts)

        relation_features = helpers.normalize_relation_features(relation_features, width=width, height=height)
        job_wrapper.progress_update("SUCCESS: Calculating Features.")

        # Text Segmenting
        # text string label converter
        job_wrapper.progress_update("Text Segmentation.")
        TextSegmentsField = Field(sequential=True, use_vocab=True, include_lengths=True, batch_first=True)
        TextSegmentsField.vocab = keys_vocab_cls

        text_segments = [list(trans) for trans in transcripts[:boxes_num]]

        # texts shape is (num_texts, max_texts_len), texts_len shape is (num_texts,)
        texts, texts_len = TextSegmentsField.process(text_segments)
        texts = texts[:, :transcript_len].numpy()
        texts_len = np.clip(texts_len.numpy(), 0, transcript_len)
        text_segments = (texts, texts_len)

        # Adjusting the Mask
        for i in range(boxes_num):
            mask[i, :texts_len[i]] = 1

        job_wrapper.progress_update("SUCCESS: Text Segmentation.")

        # Document Output Parameters
        job_wrapper.progress_update("Coding Information.")
        whole_image = RawField().preprocess(image)
        text_segments = TextSegmentsField.preprocess(text_segments)  # (text, texts_len)
        boxes_coordinate = RawField().preprocess(resized_boxes)
        relation_features = RawField().preprocess(relation_features)
        mask = RawField().preprocess(mask)
        boxes_num = RawField().preprocess(boxes_num)
        transcript_len = RawField().preprocess(transcript_len)  # max transcript len of current document

        # Convert Document Output Parameters to Model's Format
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        max_boxes_num_batch = boxes_num
        max_transcript_len = transcript_len

        image_batch_tensor = torch.stack([trsfm(whole_image)], dim=0).float()

        relation_features_padded_list = [F.pad(torch.FloatTensor(relation_features), (0, 0, 0, max_boxes_num_batch - boxes_num, 0, max_boxes_num_batch - boxes_num))]
        relation_features_batch_tensor = torch.stack(relation_features_padded_list, dim=0)

        boxes_coordinate_padded_list = [F.pad(torch.FloatTensor(boxes_coordinate), (0, 0, 0, max_boxes_num_batch - boxes_num))]
        boxes_coordinate_batch_tensor = torch.stack(boxes_coordinate_padded_list, dim=0)

        text_segments_padded_list = [F.pad(torch.LongTensor(text_segments[0]), (0, max_transcript_len - transcript_len, 0, max_boxes_num_batch - boxes_num), value=keys_vocab_cls.stoi['<pad>'])]
        text_segments_batch_tensor = torch.stack(text_segments_padded_list, dim=0)

        text_length_padded_list = [F.pad(torch.LongTensor(text_segments[1]), (0, max_boxes_num_batch - boxes_num))]
        text_length_batch_tensor = torch.stack(text_length_padded_list, dim=0)

        mask_padded_list = [F.pad(torch.ByteTensor(mask), (0, max_transcript_len - transcript_len, 0, max_boxes_num_batch - boxes_num))]
        mask_batch_tensor = torch.stack(mask_padded_list, dim=0)

        image_indexs_list = [0]
        image_indexs_tensor = torch.tensor(image_indexs_list)

        model_input = dict(whole_image=image_batch_tensor,
                           relation_features=relation_features_batch_tensor,
                           text_segments=text_segments_batch_tensor,
                           text_length=text_length_batch_tensor,
                           boxes_coordinate=boxes_coordinate_batch_tensor,
                           mask=mask_batch_tensor,
                           image_indexs=image_indexs_tensor)
        job_wrapper.progress_update("SUCCESS: Coding Information.")

        job_wrapper.progress_update("Preparing for Inference.")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_chkpt_path, map_location=device)

        model_config = checkpoint['config']
        state_dict = checkpoint['state_dict']

        pick_model = model_config.init_obj('model_arch', pick_arch_module)
        pick_model = pick_model.to(device)
        pick_model.load_state_dict(state_dict)
        pick_model.eval()
        job_wrapper.progress_update("SUCCESS: Preparing for Inference.")

        job_wrapper.progress_update("Performing Inference.")
        predict_output = {}
        with torch.no_grad():
            for key, input_value in model_input.items():
                if input_value is not None:
                    model_input[key] = input_value.to(device)

            output = pick_model(**model_input)
            logits = output['logits']
            new_mask = output['new_mask']
            image_indexs = model_input['image_indexs']  # (B,)
            text_segments = model_input['text_segments']  # (B, num_boxes, T)
            mask = model_input['mask']

            best_paths = pick_model.decoder.crf_layer.viterbi_tags(logits, mask=new_mask, logits_batch_first=True)
            predicted_tags = []
            for path, score in best_paths:
                predicted_tags.append(path)

            # convert iob index to iob string
            decoded_tags_list = iob_index_to_str(predicted_tags)
            # union text as a sequence and convert index to string
            decoded_texts_list = text_index_to_str(text_segments, mask)

            for decoded_tags, decoded_texts, image_index in zip(decoded_tags_list, decoded_texts_list, image_indexs):
                # List[ Tuple[str, Tuple[int, int]] ]
                spans = bio_tags_to_spans(decoded_tags, [])
                spans = sorted(spans, key=lambda x: x[1][0])

                entities = []  # exists one to many case
                for entity_name, range_tuple in spans:
                    entity = dict(entity_name=entity_name, text=''.join(decoded_texts[range_tuple[0]:range_tuple[1] + 1]))
                    entities.append(entity)

                for item in entities:
                    predict_output[item["entity_name"]] = item["text"]

        job_wrapper.progress_update("SUCCESS: Performing Inference.")

        job_wrapper.complete(predict_output)

    except Exception as e:
        job_wrapper.fail(str(e))
