import os
import glob
import xml.etree.ElementTree as ET
from pathlib import Path

# Paths
LABELS_DIR = Path('/home/ccne/Documents/YOLOMG/datasets/labels')

# Class mapping
classes = {'Drone': 0}

def convert_box(size, box):
    # Normalized [x_center, y_center, width, height]
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(xml_path, txt_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    out_lines = []
    
    for obj in root.iter('object'):
        difficult = obj.find('difficult')
        if difficult is not None and int(difficult.text) == 1:
            continue
            
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes[cls]
        
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
             float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert_box((w, h), b)
        
        out_lines.append(str(cls_id) + " " + " ".join([str(a) for a in bb]))
        
    with open(txt_path, 'w') as out_file:
        out_file.write('\n'.join(out_lines))

def main():
    print("Starting XML to TXT conversion...")
    xml_files = list(LABELS_DIR.rglob('*.xml'))
    print(f"Found {len(xml_files)} XML files.")
    
    converted = 0
    for xml_file in xml_files:
        txt_file = xml_file.with_suffix('.txt')
        convert_annotation(xml_file, txt_file)
        converted += 1
        if converted % 5000 == 0:
            print(f"Converted {converted}/{len(xml_files)} files...")
            
    print(f"Done! Converted {converted} files.")

if __name__ == '__main__':
    main()
