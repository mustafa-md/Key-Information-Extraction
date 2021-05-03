import glob
import json
import re
import spacy


class testModel:
    def __init__(self, model_path, output_directory, val_jsons=[], test_data_path=''):
        self.model = spacy.blank('en')
        self.model_path = model_path
        self.output_directory = output_directory
        self.val_jsons = val_jsons
        self.test_data_path = test_data_path
        self.result_dict_list = []

    def load_model(self):
        if 'ner' not in self.model.pipe_names:
            ner = self.model.create_pipe('ner')
            self.model.add_pipe(ner)
        ner = self.model.from_disk(self.model_path)
        return ner

    def test_ner(self, TEST_DATA):
        ner = self.load_model()

        for i, x in enumerate(TEST_DATA):
            # print("==============================================================")
            # print("Invoice Text is: ", x)
            doc = ner(x)
            # print("Predicted Entities are as below: ")
            dictionary = {}
            for ent in doc.ents:
                if ent.label_ not in dictionary:
                    dictionary[ent.label_] = [ent.text]
                else:
                    dictionary[ent.label_].append(ent.text)
                # print(ent.text, ent.start_char, ent.end_char, ent.label_)
            # json_object = json.dumps(dictionary, indent=4, ensure_ascii=False)
            self.result_dict_list.append(dictionary)
            # with open(self.output_directory + self.val_jsons[i][self.val_jsons[i].rfind('\\'):], "w") as outfile:
            #     outfile.write(json_object)
            #     outfile.close()

    def format_json_files(self):
        cnt_ent = 0
        # jsonFilesList = glob.glob(self.output_directory + '/' + '*.json')
        date_regex = r'(?:\d{1,2}[-/th|st|nd|rd\s]*)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep' \
                     r'|Oct|Nov|Dec)?[a-z\s,.]*(?:\d{1,2}[-/th|st|nd|rd)\s,]*)+(?:\d{2,4})+'

        for indx, json_data in enumerate(self.result_dict_list):
            # f_json = open(jsonFile, )
            # json_data = json.load(json_data)
            # f_json.close()

            if 'total' in json_data:

                list_total = json_data['total']

                # Just one total amount predicted
                if len(list_total) == 1:
                    # Checking if any space seperated amounts detected by model
                    split_total = list_total[0].split()
                    if len(split_total) > 1:
                        for split in split_total:
                            # Keep only strings with digits present
                            if bool(re.search(r'\d', split)): list_total = [split]
                    json_data['total'] = list_total[0]

                # More than one total amount predicted
                else:
                    # Find the maximum total amount
                    max_total_digit = float('-inf')
                    max_total = ""
                    for total in list_total:
                        total = re.sub("\D", "", total)
                        if not total: continue
                        total_digit = int(total)
                        if total_digit > max_total_digit:
                            max_total = total
                            max_total_digit = total_digit
                    json_data['total'] = max_total

                json_data['total'] = re.sub("[^0-9\.]+", "", json_data['total'])
                if not json_data['total']: del json_data['total']

            if 'address' in json_data:
                json_data['address'] = json_data['address'][-1]

            if 'company' in json_data:
                json_data['company'] = json_data['company'][-1]

            if 'date' in json_data:
                json_data['date'] = json_data['date'][-1]
                date_match = re.search(date_regex, json_data['date'])
                if date_match: json_data['date'] = json_data['date'][date_match.start():date_match.end()]
                if not json_data['date']: del json_data['date']

            cnt_ent += len(json_data.keys())
            self.result_dict_list[indx] = json_data
            # json_object = json.dumps(json_data, indent=4)

            # with open(jsonFile, "w") as outfile:
            #     outfile.write(json_object)
            #     outfile.close()

        # print("Total Entities: ", len(self.result_dict_list) * 4)
        # print("Recognized entities: ", cnt_ent)