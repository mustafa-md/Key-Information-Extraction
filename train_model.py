import os
import glob
import re
import json
import random
import matplotlib.pyplot as plt
from itertools import chain

import spacy
import numpy as np
from nltk.tokenize.punkt import PunktSentenceTokenizer
from matplotlib.ticker import MaxNLocator


class trainModel:
    def __init__(self, train_folder_path):
        self.train_folder_path = train_folder_path
        self.tokenizer = PunktSentenceTokenizer()
        self.train_data = []
        self.model = spacy.blank('en')
        self.test_f1scores = []
        self.load_data(self.train_folder_path)

    def load_data(self, data_dir):
        missed_ent = 0
        sub_directories = [os.path.join(data_dir, o) for o in os.listdir(data_dir)
                           if os.path.isdir(os.path.join(data_dir, o))]

        for folder in sub_directories:
            sub_folder = folder + '\\'
            txtFile = glob.glob(sub_folder + '*.txt')[0]
            # print(txtFile)

            with open(txtFile, "r") as textfile:
                lines = textfile.readlines()
                new_line = ""
                for line in lines:
                    if line is not None:
                        temp = line.split(',')
                        temp = [temp[i].strip() for i in range(len(temp))]
                        new_line += " ".join(temp[8:]) + " "

                sentences = self.tokenizer.tokenize(new_line)
                sentences = [s[:-1] if s.endswith(".") else s for s in sentences]
                new_line = " ".join(sentences)
                new_line = new_line.replace('(', '')
                new_line = new_line.replace(')', '')
                new_line = new_line.replace(',', '')

                new_line = re.sub(' +', ' ', new_line)

                jsonFile = glob.glob(sub_folder + '*.json')[0]
                f_json = open(jsonFile, )
                json_data = json.load(f_json)

                entity_list = []

                for j_data in json_data.keys():
                    og_value = json_data[j_data]
                    value = og_value.strip()
                    sentences = self.tokenizer.tokenize(value)
                    sentences = [s[:-1] if s.endswith(".") else s for s in sentences]
                    value = " ".join(sentences)
                    value = value.replace(',', ' ')
                    value = re.sub(' +', ' ', value)

                    if j_data == 'total' and ('$' in value or '.' in value):
                        value = re.escape(value)

                    match = (re.search(value, new_line))

                    if match is None:
                        missed_ent += 1
                        continue

                    start = match.start()
                    end = match.end()
                    entity_list.append((start, end, j_data))
                if entity_list:
                    self.train_data.append((new_line, {'entities': entity_list}))

    def train_spacy(self, TRAIN_DATA, TEST_DATA, iterations=100, dropout=0.5):
        spacy.require_gpu()

        # create the built-in pipeline components and add them to the pipeline
        # nlp.create_pipe works for built-ins that are registered with spaCy
        if 'ner' not in self.model.pipe_names:
            ner = self.model.create_pipe('ner')
            self.model.add_pipe(ner, last=True)
        else:
            ner = self.model.get_pipe("ner")

        # add labels
        label_set = set()

        for sentence, annotation in TRAIN_DATA:
            for ent in annotation.get('entities'):
                label_set.add(ent[2])

        for label in label_set: ner.add_label(label)

        # get names of other pipes to disable them during training
        other_pipes = [pipe for pipe in self.model.pipe_names if pipe != 'ner']
        with self.model.disable_pipes(*other_pipes):  # only train NER
            optimizer = self.model.begin_training(device=0)
            for itr in range(iterations):
                random.shuffle(TRAIN_DATA)
                losses = {}
                for text, annotations in TRAIN_DATA:
                    self.model.update(
                        [text],  # batch of texts
                        [annotations],  # batch of annotations
                        drop=dropout,  # dropout - make it harder to memorise data
                        sgd=optimizer,  # callable to update weights
                        losses=losses)

                print('=======================================')
                print('Interation = ' + str(itr))
                print('Losses = ' + str(losses))

                print('===============TEST DATA========================')
                scores = self.evaluate(self.model, TEST_DATA)
                self.test_f1scores.append(scores["textcat_f"])
                print('F1-score = ' + str(scores["textcat_f"]))
                print('Precision = ' + str(scores["textcat_p"]))
                print('Recall = ' + str(scores["textcat_r"]))
                print('=================================================')

    def calc_precision(self, pred, true):
        precision = len([x for x in pred if x in true]) / (len(pred) + 1e-20)  # true positives / total pred
        return precision

    def calc_recall(self, pred, true):
        recall = len([x for x in true if x in pred]) / (len(true) + 1e-20)  # true positives / total test
        return recall

    def calc_f1(self, precision, recall):
        f1 = 2 * ((precision * recall) / (precision + recall + 1e-20))
        return f1

    def evaluate(self, ner, data):
        preds = [ner(x[0]) for x in data]

        precisions, recalls, f1s = [], [], []

        # iterate over predictions and test data and calculate precision, recall, and F1-score
        for pred, true in zip(preds, data):
            true = [x[2] for x in
                    list(chain.from_iterable(true[1].values()))]  # x[2] = annotation, true[1] = (start, end, annot)
            pred = [i.label_ for i in pred.ents]  # i.label_ = annotation label, pred.ents = list of annotations
            precision = self.calc_precision(true, pred)
            precisions.append(precision)
            recall = self.calc_recall(true, pred)
            recalls.append(recall)
            f1s.append(self.calc_f1(precision, recall))

        return {"textcat_p": np.mean(precisions), "textcat_r": np.mean(recalls), "textcat_f": np.mean(f1s)}

    def create_train_test_splits(self, data):
        random.shuffle(data)
        # Creating a 80:20 split of data
        train_size = int(0.8 * len(data))
        train_split = data[:train_size]
        test_split = data[train_size:]
        return train_split, test_split

    def plot_graph_of_f1_score(self):
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.plot(self.test_f1scores, label="Test F1_score")
        ax.set_xlabel('Iterations')
        ax.set_ylabel('F1 score')
        ax.legend()
        ax.set_title('F1 score vs Iterations for test data')
        plt.show()

if __name__ == "__main__":
    train_path = 'C:\\Users\\musta\\Downloads\\Data Science\\DL Course\\imagetotext\\dataset\\train'
    model_path = 'C:\\Users\\musta\\Downloads\\Data Science\\DL Course\\imagetotext\\trained_model'
    print("Loading train data files.........")
    train_model_obj = trainModel(train_path)  # object initialization
    print("Train Data created...")
    print("Making Train Test Splits.........")
    train_split, test_split = train_model_obj.create_train_test_splits(train_model_obj.train_data)
    print("Training model.........")
    train_model_obj.train_spacy(train_split, test_split, iterations=100, dropout=0.5)
    train_model_obj.model.to_disk(model_path)
    print("Saved the trained model to ", model_path)
    train_model_obj.plot_graph_of_f1_score()
