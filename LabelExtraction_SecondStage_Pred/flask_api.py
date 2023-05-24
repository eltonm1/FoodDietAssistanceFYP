from pred_secondstage import pred
from flask import request, jsonify, Flask
from io import BytesIO
from difflib import get_close_matches
import re
from chatbot.chatbot import ChatBot
from chatbot import response
import requests

app = Flask(__name__)

nutritionInformationExtractor = pred.NutritionLabelInformationExtractor()
chatbot = ChatBot()
BACKEND_URL = "http://20.205.60.6:8000"
@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

token_list = ["energy", "protein", "total_fat", "saturated_fat", "trans_fat", "carbohydrates", "sugars", "sodium"]
@app.route("/predict", methods=["POST"])
def predict():
    # x = request
    file = request.files['formName']
    # convert that to bytes
    img_bytes = BytesIO(file.read())
    predicted_result = nutritionInformationExtractor.predict(img_bytes)
    final_result = {
        "energy": 0,
        "protein": 0.0,
        "total_fat": 0.0,
        "saturated_fat": 00.0,
        "trans_fat": 0.0,
        "carbohydrates": 0.0,
        "sugars": 0.0,
        "sodium": 0.0 #Sodium mg all other grams
    }

    for key, value in predicted_result.items():
        closest_key = get_close_matches(key.lower(), token_list, n=1, cutoff=0.6)
        if len(closest_key) > 0:
            corrected_value = re.findall(r"[-+]?(?:\d*\.*\d+)", value)
            if not corrected_value:
                corrected_value = [0]
            final_result[closest_key[0]] = float(corrected_value[0])
        final_result["energy"] = int(final_result["energy"])
    return jsonify(final_result)

    # class_id, class_name = get_prediction(image_bytes=img_bytes)
@app.route("/chatbot", methods=["POST"])
def chatbot_process():
    content = request.get_json()
    client_input = content['client_input']
    prev_prod_entity = content['prod_entity']
    (intent_id, intent_str, prod_entity, date_entity) = chatbot.process(client_input)
    response_func = getattr(response, intent_str)
    
    # r = None
    # if prod_entity:
    #     r = requests.get(f"{BACKEND_URL}/api/foodproducts/search/{prod_entity}")
    #     r = r.json()
    ret_content = response_func(client_input=client_input, prev_prod_entity=prev_prod_entity,
                   intent_id=intent_id, prod_entity=prod_entity, date_entity=date_entity)
    return jsonify(ret_content)


#flask --app flask_api run --host=0.0.0.0 --port 9999