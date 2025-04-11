'''
script to extract queue data (s3 images path/annotation information) as DFs
from snowflake and display them with their bounding box annotations 
'''

import cv2
import pandas as pd
import numpy as np
import boto3
from snowbyte import snowflake_query_to_df
from functools import partial
from collections import defaultdict


def get_snow_queues(queue_names, ssm_name = "/dsn/snowflake/anthony"):

    snowflake_query = partial(snowflake_query_to_df, ssm_name=ssm_name)

    sql=f"""   
                    select
                        pw.id as pw_id
                        , pw.name as workflow_name
                        , pi.id as pi_id
                        , pi.metadata
                        , pi.images as image_urls
                        , pa.id as pa_id
                        , pa.annotation
                        , pa.annotator_email
                        , pa.plali_image_id
                        , pa.id as plali_annotation_id
                        , pa.annotation_time as annotation_time
                        , pa.duration as duration

                    from     prod.plali_workflows as pw
                        join prod.plali_images as pi on pi.workflow_id = pw.id
                        join prod.plali_annotations as pa on pa.plali_image_id = pi.id

                    where workflow_name in {queue_names};
                """
    df = snowflake_query(sql, json_cols=["image_urls", "annotation"])

    anns=flatten_annotations(df)
    
    info = {} 
    for label, label_df in anns.groupby('label'):
        info[label] = f"Num annotations {len(label_df)}"

    return df, anns, info

def flatten_annotations(df):
    '''
    extract annotations from df (used within get_snow_queues)
    '''
    anns=defaultdict(list)
    for i, r in df.iterrows():
        if r.annotation:
            ann = r.annotation
            if "annotations" not in ann:
                continue
        else:
            continue

        for a in ann["annotations"]:
            anns["image_url"].append(r.image_urls[0]    )
            anns["annotator_email"].append(r.annotator_email)
            anns["label"].append(a.get("label", None))
            anns["category"].append(a.get("category", None))
            anns["height"].append(a.get("height", None))
            anns["width"].append(a.get("width", None))
            anns["xCrop"].append(a.get("xCrop", None))
            anns["yCrop"].append(a.get("yCrop", None))
            anns["workflow_name"].append(r.workflow_name)
            anns["plali_annotation_id"].append(r.plali_annotation_id)
            anns["plali_image_id"].append(r.plali_image_id)
            anns["skipped"].append(False)
            anns["duration"].append(pd.Timedelta(r.duration).total_seconds())
            anns["annotation_time"].append(r.annotation_time)

    return pd.DataFrame(anns)

def get_img_from_s3(image_url, bucket = "aquabyte-datasets-images"):
    s3_resource = boto3.resource('s3')
    bucket_crops = s3_resource.Bucket(bucket)
    key=image_url.split("s3://aquabyte-datasets-images/")[1]
    try:
        img = bucket_crops.Object(key).get().get('Body').read()
    except:
        print("failed to get;", key)
        return
    ima = cv2.imdecode(np.asarray(bytearray(img)), cv2.IMREAD_COLOR)
    
    return ima

def extract_imgs_with_bbox(df, mapping = {"CLEAR_PELLET" : (0, 255, 0), 'PELLET' : (0, 0, 255)}, n = None, skip_empty = False):
    imgs = []
    bbox_imgs = []

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    i = 0
    for _, row in df.iterrows():
        
        bboxes = row['annotation']['annotations']

        if skip_empty is True:
            if len(bboxes) == 0: 
                continue
         
        img = get_img_from_s3(row['image_urls'][0])
        if img is None:
            continue
        
        i += 1

        image_with_bbox = draw_bboxes_with_labels(np.copy(img), bboxes, mapping = mapping)

        imgs.append(img)
        bbox_imgs.append(image_with_bbox)
        
        if n is not None:
            if i == n:
                break 

        
    return imgs, bbox_imgs

def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=1,
          font_thickness=1,
          text_color=(0, 255, 0),
          ):
    '''
    Draws a text string on an image at a specified position.

    Parameters:
        img (ndarray): The image on which the text will be drawn.
        text (str): The text string to be drawn.
        font (int, optional): The font type to use. Default is cv2.FONT_HERSHEY_PLAIN.
        pos (tuple, optional): The position (x, y) where the text will start. Default is (0, 0).
        font_scale (float, optional): The scale of the font. Default is 1.
        font_thickness (int, optional): The thickness of the text characters. Default is 1.
        text_color (tuple, optional): The color of the text in BGR format. Default is green (0, 255, 0).

    Returns:
        ndarray: The image with the text drawn on it.
    '''

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    img = cv2.putText(img, text, (x, y - 5 + font_scale - 5), font, font_scale, text_color, font_thickness)

    return img

def draw_bboxes_with_labels(img, bboxes, mapping = {"CLEAR_PELLET" : (0, 255, 0), 'PELLET' : (0, 0, 255)}):
    '''
    Draws bounding boxes with labels on an image.

    Parameters:
        img (ndarray): The image on which the bounding boxes and labels will be drawn.
        bboxes (list of dicts): A list of bounding box dictionaries, each containing the keys 'xCrop', 'yCrop', 'width', 'height', and 'label'.
        mapping (dict, optional): A dictionary that maps labels to colors. Default is {"CLEAR_PELLET": (0, 255, 0), 'PELLET': (0, 0, 255)}.

    Returns:
        ndarray: The image with the bounding boxes and labels drawn.
    '''
    #apply bboxes
    for box in bboxes:

        if box['label'] in mapping:
            c = mapping[box['label']]
        
        else:
            c = (255, 255, 255) # Default color is black if label not found in mapping

        top_left = (int(box['xCrop']), int(box['yCrop']))
        bottom_right = (int(box['xCrop'] + box['width']), int(box['yCrop'] + box['height']))
        
        img = cv2.rectangle(img, top_left, bottom_right, c, 1)
        img = draw_text(img, box['label'], pos = top_left, text_color=c)

    return img

def draw_bboxes(img, box):

    '''
    Draws a single bounding box on an image.

    Parameters:
        img (ndarray): The image on which the bounding box will be drawn.
        box (dict): A dictionary containing the keys 'xCrop', 'yCrop', 'width', and 'height' representing the bounding box coordinates and size.

    Returns:
        ndarray: The image with the bounding box drawn.
    '''        

    top_left = (int(box['xCrop']), int(box['yCrop']))
    bottom_right = (int(box['xCrop'] + box['width']), int(box['yCrop'] + box['height']))
    
    image_with_bbox = cv2.rectangle(img, top_left, bottom_right, (0,0, 255), 1)

    return image_with_bbox



if __name__ == '__main__':
    queue_names="""('pellet_bbox_aquanvr_feeding_cam_2025-03-01_2025-03-15_q1')"""
    df, anns, info = get_snow_queues(queue_names)
    print('extracted DF from snowflake')
    img, bbox_images = extract_imgs_with_bbox(df, n = 2, skip_empty=True)
    print(f'extracted {len(img)} images')

    cv2.imwrite('img1.png', bbox_images[0])
    cv2.imwrite('img2.png', bbox_images[1])