import os
import base64

import flask_monitoringdashboard as dashboard
from wsgiref import simple_server
from flask import Flask, request, Response, render_template, jsonify
from flask_cors import CORS, cross_origin

from train_model import trainModel
from test_model import testModel
from ocr import Ocr

application = Flask(__name__)
dashboard.bind(application)
CORS(application)

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')
CORS(application)

train_path = 'C:\\Users\\musta\\Downloads\\Data Science\\DL Course\\imagetotext\\dataset\\train'
model_path = 'trained_model'
output_directory = 'out'

imagePath = "images/inputImage.jpg"


def decodeImageIntoBase64(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(fileName):
    with open(fileName, "rb") as f:
        return base64.b64encode(f.read())


@application.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@application.route("/predict", methods=["POST"])
@cross_origin()
def getPrediction():
    try:
        if 'image' in request.json and request.json['image'] is not None:
            inpImage = request.json['image']
            decodeImageIntoBase64(inpImage, imagePath)
        else:
            return Response("Please provide an image")

        OCR = Ocr()
        data = OCR.process_single_image(imagePath)
        test_model_obj = testModel(model_path, output_directory)
        test_model_obj.test_ner(data)
        test_model_obj.format_json_files()
        result = test_model_obj.result_dict_list[0]
        # jsonStr = json.dumps(test_model_obj.result_dict_list[0], ensure_ascii=False)
        return jsonify({"Result": result})

    except ValueError:
        return jsonify({"Result": "Error Occurred! %s" % ValueError})
        # return Response("Error Occurred! %s" % ValueError)

    except KeyError:
        return jsonify({"Result": "Error Occurred! %s" % KeyError})
        # return Response("Error Occurred! %s" % KeyError)

    except Exception as e:
        return jsonify({"Result": "Error Occurred! %s" % e})
        # return Response("Error Occurred! %s" % e)


@application.route("/train", methods=['POST'])
@cross_origin()
def trainRouteClient():
    try:
        if 'trainfolder' in request.json and request.json['trainfolder'] is not None:
            path = request.json['trainfolder']
        else:
            path = train_path
        train_model_obj = trainModel(path) #object initialization
        train_split, test_split = train_model_obj.create_train_test_splits(train_model_obj.train_data)
        train_model_obj.train_spacy(train_split, test_split, iterations=100, dropout=0.5)
        train_model_obj.model.to_disk(model_path)
        print("Saved the trained model to ", model_path)

    except ValueError:
        return jsonify({"Result": "Error Occurred! %s" % ValueError})
        # return Response("Error Occurred! %s" % ValueError)

    except KeyError:
        return jsonify({"Result": "Error Occurred! %s" % KeyError})
        # return Response("Error Occurred! %s" % KeyError)

    except Exception as e:
        return jsonify({"Result": "Error Occurred! %s" % e})
        # return Response("Error Occurred! %s" % e)

    return jsonify({"Result": "Training successfull!!"})
    # return Response("Training successfull!!")


port = int(os.getenv("PORT", 5000))
if __name__ == '__main__':
    host = '0.0.0.0'
    httpd = simple_server.make_server(host, port, application)
    # print("Serving on %s %d" % (host, port))
    httpd.serve_forever()

if __name__ == '__main__':
    application.run(debug=True)