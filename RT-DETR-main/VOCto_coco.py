import os
import json
import random
import xml.etree.ElementTree as ET
import shutil

def parse_voc_annotation(annotation_file):
    """
    解析VOC格式的XML标注文件
    :param annotation_file: XML文件路径
    :return: 图像宽度、高度、标注信息列表
    """
    tree = ET.parse(annotation_file)
    root = tree.getroot()

    # 获取图像宽度和高度
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    # 获取标注信息
    annotations = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        bbox = obj.find("bbox")
        if bbox is None:
            bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        annotations.append({
            "class_name": class_name,
            "bbox": [xmin, ymin, xmax, ymax]
        })

    return width, height, annotations

def generate_coco_json(voc_image_dir, voc_annotation_dir, categories, output_json, image_ids):
    """
    生成COCO格式的JSON文件
    :param voc_image_dir: VOC图像文件夹路径
    :param voc_annotation_dir: VOC标注文件夹路径
    :param categories: 类别列表（字典格式，{class_name: class_id}）
    :param output_json: 输出JSON文件路径
    :param image_ids: 图像ID列表
    """
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 添加类别信息
    for class_name, class_id in categories.items():
        coco_output["categories"].append({
            "id": class_id,
            "name": class_name
        })

    # 图像和标注信息
    image_id = 1
    annotation_id = 1

    for image_id in image_ids:
        annotation_filename = f"{image_id:06d}.xml"
        annotation_path = os.path.join(voc_annotation_dir, annotation_filename)
        image_filename = f"{image_id:06d}.jpg"
        image_path = os.path.join(voc_image_dir, image_filename)

        if not os.path.exists(image_path):
            print(f"图像文件 {image_path} 不存在，跳过...")
            continue

        # 解析VOC标注文件
        width, height, annotations = parse_voc_annotation(annotation_path)

        # 添加图像信息
        coco_output["images"].append({
            "id": image_id,
            "file_name": image_filename,
            "width": width,
            "height": height
        })

        # 添加标注信息
        for annotation in annotations:
            class_name = annotation["class_name"]
            class_id = categories[class_name]
            bbox = annotation["bbox"]

            coco_output["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                "iscrowd": 0
            })
            annotation_id += 1

        image_id += 1

    # 写入JSON文件
    with open(output_json, "w") as f:
        json.dump(coco_output, f, indent=4)

    print(f"生成的COCO JSON文件已保存到 {output_json}")

def split_voc_dataset(voc_annotation_dir, train_ratio=0.8):
    """
    将VOC数据集划分为训练集和验证集
    :param voc_annotation_dir: VOC标注文件夹路径
    :param train_ratio: 训练集比例
    :return: 训练集图像ID列表和验证集图像ID列表
    """
    annotation_files = [os.path.splitext(f)[0] for f in os.listdir(voc_annotation_dir) if f.endswith(".xml")]
    image_ids = [int(f) for f in annotation_files]

    random.shuffle(image_ids)
    split_index = int(len(image_ids) * train_ratio)

    train_ids = image_ids[:split_index]
    val_ids = image_ids[split_index:]

    return train_ids, val_ids

def copy_images_to_folders(voc_image_dir, train_ids, val_ids, train_dir, val_dir):
    """
    将图像文件复制到训练集和验证集文件夹
    :param voc_image_dir: VOC图像文件夹路径
    :param train_ids: 训练集图像ID列表
    :param val_ids: 验证集图像ID列表
    :param train_dir: 训练集目标文件夹
    :param val_dir: 验证集目标文件夹
    """
    # 创建目标文件夹
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 复制训练集图像
    for image_id in train_ids:
        image_filename = f"{image_id:06d}.jpg"
        source_path = os.path.join(voc_image_dir, image_filename)
        target_path = os.path.join(train_dir, image_filename)
        shutil.copy(source_path, target_path)

    # 复制验证集图像
    for image_id in val_ids:
        image_filename = f"{image_id:06d}.jpg"
        source_path = os.path.join(voc_image_dir, image_filename)
        target_path = os.path.join(val_dir, image_filename)
        shutil.copy(source_path, target_path)

    print(f"图像已复制到 {train_dir} 和 {val_dir}")

# 主函数
if __name__ == "__main__":
    # VOC数据集路径
    voc_image_dir = "F:\\LearnDL\\VOC2007\\JPEGImages"
    voc_annotation_dir = "F:\\LearnDL\\VOC2007\\Annotations"

    # 目标文件夹路径
    train_dir = "F:\\LearnDL\\coco\\train2017"
    val_dir = "F:\\LearnDL\\coco\\val2017"

    # 类别信息（VOC2007的类别）
    categories = {
        "aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
        "bottle": 5, "bus": 6, "car": 7, "cat": 8,
        "chair": 9, "cow": 10, "diningtable": 11, "dog": 12,
        "horse": 13, "motorbike": 14, "person": 15, "pottedplant": 16,
        "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20
    }

    # 划分数据集
    train_ids, val_ids = split_voc_dataset(voc_annotation_dir)

    # 生成训练集和验证集的COCO格式JSON文件
    generate_coco_json(voc_image_dir, voc_annotation_dir, categories, "F:\\LearnDL\\coco\\annotations\\instances_train2017.json", train_ids)
    generate_coco_json(voc_image_dir, voc_annotation_dir, categories, "F:\\LearnDL\\coco\\annotations\\instances_val2017.json", val_ids)

    # 将图像复制到训练集和验证集文件夹
    copy_images_to_folders(voc_image_dir, train_ids, val_ids, train_dir, val_dir)