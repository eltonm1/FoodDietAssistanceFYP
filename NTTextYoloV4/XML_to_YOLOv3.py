#================================================================
#
#   File name   : XML_to_YOLOv3.py
#   Author      : PyLessons
#   Created date: 2020-06-04
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : used to convert XML labels to YOLOv3 training labels
#
#================================================================
import xml.etree.ElementTree as ET
import os
import glob
import config

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "test-(to-be-train)")
annotation_txt_path = os.path.join(data_dir, "annotation.txt")
classes_txt_path = os.path.join(data_dir, "classes.txt")
classes = []
      
def ParseXML(img_folder, file):
    for xml_file in glob.glob(img_folder+'/*.xml'):
        tree=ET.parse(open(xml_file))
        root = tree.getroot()
        image_name = root.find('filename').text
        img_path = img_folder+'/'+image_name
        for i, obj in enumerate(root.iter('object')):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes:
                classes.append(cls)
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            OBJECT = (str(int(float(xmlbox.find('xmin').text)))+','
                      +str(int(float(xmlbox.find('ymin').text)))+','
                      +str(int(float(xmlbox.find('xmax').text)))+','
                      +str(int(float(xmlbox.find('ymax').text)))+','
                      +str(cls_id))
            img_path += ' '+OBJECT
        file.write(img_path+'\n')

def run_XML_to_YOLOv3(): 
    with open(annotation_txt_path, "w") as file:
        print(annotation_txt_path)
        ParseXML(data_dir, file)

    print("Classes:", classes)
    with open(classes_txt_path, "w") as file:
        for name in classes:
            file.write(str(name)+'\n')

run_XML_to_YOLOv3()
