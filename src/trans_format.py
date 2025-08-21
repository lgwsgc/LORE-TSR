import os
import json

coco = dict()
coco['images'] = []
coco['type'] = 'instances'
coco['annotations'] = []
coco['categories'] = []

category_set = dict()
image_set = set()

category_item_id = 0
image_id = 20140000000
annotation_id = 0


def addCatItem(name):
    global category_item_id
    category_item = dict()
    category_item['supercategory'] = 'none'
    category_item_id += 1
    category_item['id'] = category_item_id
    category_item['name'] = name
    coco['categories'].append(category_item)
    category_set[name] = category_item_id
    return category_item_id


def addImgItem(coco,file_name, size):
    global image_id
    if file_name is None:
        raise Exception('Could not find filename tag in xml file.')
    if size['width'] is None:
        raise Exception('Could not find width tag in xml file.')
    if size['height'] is None:
        raise Exception('Could not find height tag in xml file.')
    image_id += 1
    image_item = dict()
    image_item['id'] = image_id
    image_item['file_name'] = file_name
    image_item['width'] = size['width']
    image_item['height'] = size['height']
    coco['images'].append(image_item)
    image_set.add(file_name)
    return image_id


def addAnnoItem(coco,image_id, category_id, bbox, seg,cell):
    global annotation_id
    annotation_item = dict()
    annotation_item['segmentation'] = []
    annotation_item["logic_axis"] = [[cell[1],cell[3],cell[0],cell[2]]]
    annotation_item['segmentation'].append(seg)
    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    annotation_id += 1
    annotation_item['id'] = annotation_id
    coco['annotations'].append(annotation_item)


def parseJsonFiles(data):
        coco = dict()
        coco['images'] = []
        coco['annotations'] = []
        coco['categories'] = []
        i = 0
        for k, v in data.items():
            # i+=1
            # if i%5!=0:
            #     continue
            image_dict = dict()
            image_dict['width'] = v['width']
            image_dict['height'] = v['height']
            current_image_id = addImgItem(coco,k, image_dict)
            current_category_id = 1
            bboxes = v['content_ann']['bboxes']
            cells = v['content_ann']['cells']
            for box, cell in zip (bboxes,cells):
                if not len(box):
                    continue
                bbox = [box[0],box[1],box[2]-box[0],box[3]-box[1]]
                seg = [box[0],box[1],box[2],box[1],box[2],box[3],box[0],box[3]]
                addAnnoItem(coco,current_image_id, current_category_id, bbox, seg,cell)
        return coco
if __name__ == '__main__':
    lgpma_json_path = r"/ssdata/user/zxy/data/PTN/train_data_second_3573.json"
    lore_json_path = r"/ssdata/user/gangwei.li/Code/LORE-TSR1/data/elect-table-331/labels/train.json"
    # f = open(lgpma_json_path, "r", encoding="utf-8")
    # data = json.load(f)
    # coco_data  = parseJsonFiles(data)
    # f_w = open(lore_json_path, "w", encoding="utf-8")
    # json.dump(coco_data,f_w)

    f_w = open(lore_json_path, "r", encoding="utf-8")
    coco_data = json.load(f_w)
    for s in coco_data['annotations']:
        print(s)