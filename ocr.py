import tesserocr
import re
import os
from PIL import Image
import glob
from nltk.tokenize.punkt import PunktSentenceTokenizer


def pre_process_sentences(sentences, tokenizer):
    new_line = " ".join(sentences)

    sentences = tokenizer.tokenize(new_line)
    sentences = [s[:-1] if s.endswith(".") else s for s in sentences]
    new_line = " ".join(sentences)
    new_line = re.sub("[^0-9a-zA-Z\-\'\.\:\/]+", " ", new_line)
    new_line = new_line.replace(',', '')
    new_line = re.sub(' +', ' ', new_line)
    extra_char_index = new_line.find("**")
    if extra_char_index != -1: new_line = new_line[:extra_char_index]
    extra_char_index = new_line.find("Thank")
    if extra_char_index != -1: new_line = new_line[:extra_char_index]

    return new_line


class Ocr:
    def __init__(self):
        # self.data_dir = data_dir
        self.tokenizer = PunktSentenceTokenizer()
        # self.val_jsons = []

    def process_single_image(self, image_path):
        Data = []
        image_text = tesserocr.image_to_text(Image.open(image_path))
        sentences = image_text.split('\n')
        sentences = [line.strip() for line in sentences]
        sentences = [line for line in sentences if line]
        Data.append(pre_process_sentences(sentences, self.tokenizer))
        # self.val_jsons.append(image_path[:-3] + 'json')
        return Data

    def process_batch(self, data_dir):
        sub_directories = [os.path.join(data_dir, o) for o in os.listdir(data_dir)
                           if os.path.isdir(os.path.join(data_dir, o))]

        Data = []
        for folder in sub_directories:
            sub_folder = folder + '/'
            image_path = glob.glob(sub_folder + '*.jpg')[0]
            image_text = tesserocr.image_to_text(Image.open(image_path))
            sentences = image_text.split('\n')
            sentences = [line.strip() for line in sentences]
            sentences = [line for line in sentences if line]
            Data.append(pre_process_sentences(sentences, self.tokenizer))
            # self.val_jsons.append(image_path[:-3] + 'json')

        return Data
