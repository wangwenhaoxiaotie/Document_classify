import logging
import os
import csv
import json
from PIL import Image, ImageFile
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

logger = logging.getLogger(__name__)


class OcrDataset(Dataset):
    def __init__(self, data_dir, tokenizer, labels, mode, max_seq_length=512):
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            data_dir,
            "cached_{}_{}_v2".format(
                mode,
                str(max_seq_length),
            ),
        )
        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            logger.info("Creating features from dataset file at %s", data_dir)
            examples = read_examples_from_file(data_dir, mode)
            features = convert_examples_to_features(
                examples,
                labels,
                max_seq_length,
                tokenizer,
                cls_token=tokenizer.cls_token,
                sep_token=tokenizer.sep_token,
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
            )
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

        self.features = features
        #import ipdb;ipdb.set_trace()
        # Convert to Tensors and build dataset
        self.all_input_ids = torch.tensor(
            [f.input_ids for f in features], dtype=torch.long
        )
        self.all_bboxes = torch.tensor(
            [f.boxes for f in features], dtype=torch.long
        )
        self.all_input_mask = torch.tensor(
            [f.input_mask for f in features], dtype=torch.long
        )
        self.all_segment_ids = torch.tensor(
            [f.segment_ids for f in features], dtype=torch.long
        )        
        self.all_label_id = torch.tensor(
            [f.label_id for f in features], dtype=torch.long
        )
        self.image_transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.all_file_names = [f.file_name for f in features]       

    def __len__(self):
        return len(self.features) 

    def __getitem__(self, index):
        #print(self.features[index].file_name)
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image = Image.open(self.features[index].file_name).convert('RGB')
        image = self.image_transform(image)

        return dict(
            input_ids=self.all_input_ids[index],
            token_type_ids=self.all_segment_ids[index],
            bbox=self.all_bboxes[index],
            attention_mask=self.all_input_mask[index],
            image=image,
            label=self.all_label_id[index],
            file_names=self.all_file_names[index]
        )

class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, label, boxes, actual_bboxes, file_name, page_size):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.label = label
        self.boxes = boxes
        self.actual_bboxes = actual_bboxes
        self.file_name = file_name
        self.page_size = page_size

    def __repr__(self):
        s = ""
        for k in ["guid", "words", "label", "boxes", "actual_bboxes", "file_name", "page_size"]:
            s += f"{k}: {self.__dict__[k]}\n"
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        input_ids,
        input_mask,
        segment_ids,
        label_id,
        boxes,
        actual_bboxes,
        file_name,
        page_size,
    ):
        assert (
            0 <= all(boxes) <= 1000
        ), "Error with input bbox ({}): the coordinate value is not between 0 and 1000".format(
            boxes
        )
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.boxes = boxes
        self.actual_bboxes=actual_bboxes
        self.file_name = file_name
        self.page_size = page_size

    def __repr__(self):
        s = ""
        for k in ["input_ids", "input_mask", "label_id", "boxes", "file_name", "page_size"]:
            s += f"{k}: {self.__dict__[k]}\n"
        return s


def read_examples_from_file(data_dir, mode, threshold=0.5):
    csv_file = os.path.join(data_dir, "{}.csv".format(mode))
    guid_index = 1
    examples = []
    with open(csv_file, "r", encoding="utf-8") as fc:
        csv_reader = csv.reader(fc)
        for img_path, json_path, label in tqdm(csv_reader):
            img_path = os.path.join(data_dir, img_path)
            json_path = os.path.join(data_dir, json_path)
           
            #json_data = json.load(open(json_path, "r", encoding="utf-8"))["result"]
            json_data = json.load(open(json_path, "r", encoding="utf-8"))['ret_data']['details'][0]
            width, height = json_data["width"], json_data["height"]
            words = []
            boxes = []
            actual_bboxes = []
            file_name = img_path
            page_size = (width, height)

            for item in json_data["ocr_contents"]:
                if item.get("type") == "textbox" and item.get("ocr_confidence", 0) > threshold:
                    words.append(item.get("text", ""))
                    x1, y1, w, h = item.get("rect")
                    x2, y2 = x1 + w, y1 + h
                    boxes.append([int(round(1000 * x1 / width)), int(round(1000 * y1 / height)),
                        int(round(1000 * x2 / width)), int(round(1000 * y2 / height))])
                    actual_bboxes.append([x1, y1, x2, y2])

            examples.append(
                InputExample(
                    guid="{}-{}".format(mode, guid_index),
                    words=words,
                    label=label,
                    boxes=boxes,
                    actual_bboxes=actual_bboxes,
                    file_name=file_name,
                    page_size=page_size,
                )
            )
            guid_index += 1
    return examples


def convert_examples_to_features(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    cls_token_box=[0, 0, 0, 0],
    sep_token_box=[1000, 1000, 1000, 1000],
    pad_token_box=[0, 0, 0, 0],
    mask_padding_with_zero=True,
):
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm(enumerate(examples)):
        file_name = example.file_name
        page_size = example.page_size
        width, height = page_size
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        token_boxes = []
        actual_bboxes = []
        label_id = label_map[example.label]
        for word, box, actual_bbox in zip(
            example.words, example.boxes, example.actual_bboxes
        ):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            token_boxes.extend([box] * len(word_tokens))
            actual_bboxes.extend([actual_bbox] * len(word_tokens))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]
            actual_bboxes = actual_bboxes[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        token_boxes += [sep_token_box]
        actual_bboxes += [[0, 0, width, height]]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            token_boxes += [sep_token_box]
            actual_bboxes += [[0, 0, width, height]]

        
        segment_ids = [0] * len(tokens) 
        
        if cls_token_at_end:
            tokens += [cls_token]
            token_boxes += [cls_token_box]
            actual_bboxes += [[0, 0, width, height]]
            segment_ids += [0]
        else:
            tokens = [cls_token] + tokens
            token_boxes = [cls_token_box] + token_boxes
            actual_bboxes = [[0, 0, width, height]] + actual_bboxes
            segment_ids = [0] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = (
                [0 if mask_padding_with_zero else 1] * padding_length
            ) + input_mask
            segment_ids = ([0] * padding_length) + segment_ids
            token_boxes = ([pad_token_box] * padding_length) + token_boxes
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [0] * padding_length
            token_boxes += [pad_token_box] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(token_boxes) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("label_id: %s", label_id)
            logger.info("boxes: %s", " ".join([str(x) for x in token_boxes]))
            logger.info("actual_bboxes: %s", " ".join([str(x) for x in actual_bboxes]))
        
        #import ipdb; ipdb.set_trace()
        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
                boxes=token_boxes,
                actual_bboxes=actual_bboxes,
                file_name=file_name,
                page_size=page_size,
            )
        )
    return features


if __name__ == "__main__":
    from transformers import BertTokenizer
    #labels = open('receipt_data/labels.txt').read().split('\n')
    labels = ["医疗收据", "门诊病历", "费用清单", "处方"]
    tokenizer = BertTokenizer.from_pretrained("receipt_data/vocab.txt", do_lower_case=True)

    examples = read_examples_from_file("test_data", "test")
    features = convert_examples_to_features(
        examples,
        labels,
        512,
        tokenizer,
        cls_token=tokenizer.cls_token,
        sep_token=tokenizer.sep_token,
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    )
   
    for f in features:
        print(f)

#     from torch.utils.data import DataLoader

#     dataset = OcrDataset("receipt_data", tokenizer, labels, "train")

#     dataloader = DataLoader(dataset, batch_size=10)

#     for item in dataset:
#         for k, v in item.items():
#             print(k, v.size())
#         print(item['label'])

#     for item in dataloader:
#         for k, v in item.items():
#             print(k, v.size())
