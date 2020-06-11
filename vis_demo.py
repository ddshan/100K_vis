import os, json, glob, random
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
random.seed(3)


def ratio2coord(ratio, width, height): # str, list
    x1, y1, x2, y2 = int(float(ratio[0])*width), int(float(ratio[1])*height), int(float(ratio[2])*width), int(float(ratio[3])*height)
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, width)
    y2 = min(y2, height)
    bbox = [x1, y1, x2, y2]
    return bbox


def bbox2center(bbox):
    return (int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2))


def draw_labels(image, hand_info, hand_idx, width, height, font):
    hand_bbox_ratio = [hand_info['x1'], hand_info['y1'], hand_info['x2'], hand_info['y2']]
    hand_bbox = ratio2coord(hand_bbox_ratio, width, height)
    hand_side = hand_info['hand_side']
    contact_state = hand_info['contact_state']
    draw = ImageDraw.Draw(image)
    if hand_side == 'l':
        side_idx = 0
    elif hand_side == 'r':
        side_idx = 1

    # hand mask
    mask = Image.new('RGBA', (width, height))
    pmask = ImageDraw.Draw(mask)
    pmask.rectangle(hand_bbox, outline=hand_rgb[side_idx], width=4, fill=hand_rgba[side_idx])
    image.paste(mask, (0,0), mask)
    
    # hand text
    draw.rectangle([hand_bbox[0], max(0, hand_bbox[1]-30), hand_bbox[0]+62, max(0, hand_bbox[1]-30)+30], fill=(255, 255, 255), outline=hand_rgb[side_idx], width=4)
    draw.text((hand_bbox[0]+6, max(0, hand_bbox[1]-30)-2), f'{hand_side.upper()}-{state_map2[int(float(contact_state))]}', font=font, fill=(0,0,0)) # 


    if hand_info['obj_bbox'] is not None:

        obj_info = hand_info['obj_bbox']
        obj_bbox_ratio = [obj_info['x1'], obj_info['y1'], obj_info['x2'], obj_info['y2']]
        obj_bbox = ratio2coord(obj_bbox_ratio, width, height)
        
        # object mask
        mask = Image.new('RGBA', (width, height))
        pmask = ImageDraw.Draw(mask)
        pmask.rectangle(obj_bbox, outline=obj_rgb, width=4, fill=obj_rgba)
        image.paste(mask, (0,0), mask)

        # object text
        draw.rectangle([obj_bbox[0], max(0, obj_bbox[1]-30), obj_bbox[0]+32, max(0, obj_bbox[1]-30)+30], fill=(255, 255, 255), outline=obj_rgb, width=4)
        draw.text((obj_bbox[0]+5, max(0, obj_bbox[1]-30)-2), f'O', font=font, fill=(0,0,0)) #

        # link
        hand_center = bbox2center(hand_bbox)
        obj_center = bbox2center(obj_bbox)
        draw.line([hand_center, obj_center], fill=hand_rgb[side_idx], width=4)
        x, y = hand_center[0], hand_center[1]
        r=7
        draw.ellipse((x-r, y-r, x+r, y+r), fill=hand_rgb[side_idx])
        x, y = obj_center[0], obj_center[1]
        draw.ellipse((x-r, y-r, x+r, y+r), fill=obj_rgb)

    # elif contact_state > 0 and hand_info['obj_bbox'] is None:
    #     print(image_path)
    #     print()

    return image
    

hand_rgb = [(0, 90, 181), (220, 50, 32)] 
hand_rgba = [(0, 90, 181, 70), (220, 50, 32, 70)]

obj_rgb = (255, 194, 10)
obj_rgba = (255, 194, 10, 70)

state_map = {0:'No Contact', 1:'Self Contact', 2:'Another Person', 3:'Portable Object', 4:'Stationary Object'}
state_map2 = {0:'N', 1:'S', 2:'O', 3:'P', 4:'F'}



if __name__ == '__main__':

    # updata to your location here
    dataset_dir = '/w/dandans/Dataset_to_release/raw'
    annot_filepath = '/w/dandans/Dataset_to_release/data/train.json'
    vis_dir = './images_draw'
    os.makedirs(vis_dir, exist_ok=True)

    with open(annot_filepath, 'r') as f:
        annot_info = json.load(f)
  
    # vis randomly 
    image_list = list(annot_info.keys())
    random.shuffle(image_list)

    # data
    pbar = tqdm(enumerate(image_list))

    # draw annotation on image
    for image_idx, image_path in tqdm(enumerate(image_list)):
        image_info = annot_info[image_path]
        image = Image.open(os.path.join(dataset_dir, image_path)).convert("RGBA")
        image_name = os.path.split(image_path)[-1][:-4]
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype('./times_b.ttf', size=30)

        for hand_idx, hand_info in enumerate(image_info):

            width, height = hand_info['width'], hand_info['height']
            image = draw_labels(image, hand_info, hand_idx, width, height, font=font)

        image.save(f'{vis_dir}/{image_name}.png')

        if image_idx == 50: break

