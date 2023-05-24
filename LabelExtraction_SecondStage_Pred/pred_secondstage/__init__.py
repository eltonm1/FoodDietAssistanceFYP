# import logging

# import azure.functions as func
# from .pred import NutritionLabelInformationExtractor
# from PIL import Image
# import numpy as np

# extractor = NutritionLabelInformationExtractor()

# def main(req: func.HttpRequest) -> func.HttpResponse:
#     logging.info('Python HTTP trigger function processed a request.')

#     name = req.params.get('name')
#     #Get the image from the request
#     image = req.files.get('image')
#     logging.info(f'Get file {image}')
#     if image is not None:
#         img = Image.open(image)
#         w, h = img.size
#         output_bboxes = extractor.craft_detector.detect(np.array(img))
#         logging.info('craft_detector done.')
#         output_bboxes = sorted(output_bboxes, key=lambda x: x[1], reverse=True)
#         keys = []
#         values = []
#         for box in output_bboxes:
#             if box[0] < w/2:
#                 keys.append(box)
#             else:
#                 values.append(box)

#         keys_text = []
#         logging.info('keys detection start.')
#         for box in keys:
#             top_left_x, top_left_y, bot_right_x, bot_right_y= box
#             cropped = np.array(img)[top_left_y:bot_right_y, top_left_x:bot_right_x]
#             cropped = Image.fromarray(cropped).convert('L')
#             cropped = cropped.resize((160, 32), resample=Image.Resampling.BILINEAR)
#             text = extractor.text_recognizer.detect(np.array(cropped))
#             keys_text.append(text)
#         logging.info('keys detection done.')
#         values_text = []
#         for box in values:
#             top_left_x, top_left_y, bot_right_x, bot_right_y= box
#             cropped = np.array(img)[top_left_y:bot_right_y, top_left_x:bot_right_x]
#             cropped = Image.fromarray(cropped).convert('L')
#             cropped = cropped.resize((160, 32), resample=Image.Resampling.BILINEAR)
#             text = extractor.text_recognizer.detect(np.array(cropped))
#             values_text.append(text)
#         ket_value = dict(zip(keys_text, values_text))
#         return func.HttpResponse(f"Hello {w} {h} {ket_value}. This HTTP triggered function executed successfully.")

#     if not name:
#         try:
#             req_body = req.get_json()
#         except ValueError:
#             pass
#         else:
#             name = req_body.get('name') 

#     if name:
#         return func.HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully.")
#     else:
#         return func.HttpResponse(
#              "This HTTP triggered function executed successfully2. Pass a name in the query string or in the request body for a personalized response.",
#              status_code=200
#         )
