import os
import cv2
from itertools import chain
import base64
import pandas as pd
import requests
import json

def ocr_using_google_api(image_path, request_url):
    '''
    This function uses Google Vision API for Text Detection

    Args :
        image_path : Input image path

    Returns:
        pd.DataFrame having coordinates of each text box along with detected
        text inside it.
    '''
    lstr_filename, str_extension = os.path.splitext(str(image_path))

    image_arr = cv2.imread(image_path)
    _, image_buffer = cv2.imencode("."+str_extension,
                                            image_arr)

    int_respose_code = 0
    json_request_header = {
        'content-type': 'application/json',
        'Accept-Charset': 'UTF-8'
    }

    str_encode_image = base64.b64encode(image_buffer).decode()
    json_request_payload = {'requests':
            [
                {
                    "image":
                    {
                        'content':str_encode_image
                    },
                    'features':
                    [
                        {
                            'type': 'DOCUMENT_TEXT_DETECTION'
                        }
                    ],
                }
            ]
        }

    list_block_coordinates = []
    list_block_word_coordinates = []
    list_each_word_coordinate = []

    str_http_response = \
        requests.post(
            request_url,
            data=json.dumps(json_request_payload),
            headers=json_request_header,
            verify=False
        )

    int_respose_code = str_http_response.status_code
    if int_respose_code != 200:
        return list_block_coordinates

    else:
        json_response_data = json.loads(str_http_response.text)
        if json_response_data['responses'][0]:

            list_bounding_boxes = \
                json_response_data['responses'][0]['fullTextAnnotation']\
                    ['pages'][0]['blocks']

            list_vertices = \
                [boundingBox['boundingBox'] for boundingBox in \
                    list_bounding_boxes if 'boundingBox' in boundingBox]

            list_block_coordinates = \
                [list(chain(*[[x['x'], x['y']] for x in i['vertices']])) \
                    for i in list_vertices]

            list_block_words = []
            for bounding_box in list_bounding_boxes:
                list_paragraphs = bounding_box["paragraphs"]
                str_word = ""
                list_bounding_box = []

                for paragraphs in list_paragraphs:
                    list_words = paragraphs['words']
#                                 list_bounding_box = []
#                                 str_word = ""
                    for words in list_words:
                        list_vertices = []
                        str_text = ""
                        llst_symbols = words['symbols']
                        list_bounding_box.append(words['boundingBox'])

                        for symbols in llst_symbols:
                            str_text = (str_text + symbols['text']).strip()
                            list_vertices.append(symbols['boundingBox'])

                        str_word = (str_word + " " + str_text).strip()
                        list_word_coords = \
                            list(chain(*[[x['x'], x['y']] for x in \
                                words['boundingBox']['vertices']]))

                        list_word_coords.insert(0, str_text)
                        list_each_word_coordinate.append(list_word_coords)

                    list_word_coordinates = \
                        [list(chain(*[[x['x'], x['y']] for x in \
                            i['vertices']])) for i in list_bounding_box]

                list_block_words.append(str_word)
                list_block_word_coordinates.append(list_word_coordinates)

        for int_index, llst_block_coordinate in enumerate(list_block_coordinates):
            llst_block_coordinate.insert(0, list_block_words[int_index])

        list_word_objects = \
            [[min(item[1],item[5]), min(item[2],item[6]),
            max(item[1],item[5]), max(item[2],item[6]),
            item[0]] for i, item in enumerate(list_each_word_coordinate)]

        return df = pd.DataFrame(list_word_objects,
                            columns=['xmin', 'ymin', 'xmax', 'ymax', 'Object'])
