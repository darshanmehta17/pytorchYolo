import xml.etree.ElementTree as ET
import glob

files = sorted(glob.glob('/home/mitchell/Eyesea-download_data/Proposal ID 70294/eyesea/people/xuwe421/Delete/train_y/*.xml'))

for _file in files:
    tree = ET.parse(_file)
    root = tree.getroot()
    for child in root:
        if child.tag == 'filename':
            filename = child.text
            
        if child.tag == 'object':
            for form in root.findall("./object/bndbox/xmin"):
                x_min = form.text
            for form in root.findall("./object/bndbox/ymin"):
                y_min = form.text
            for form in root.findall("./object/bndbox/xmax"):
                x_max = form.text
            for form in root.findall("./object/bndbox/ymax"):
                y_max = form.text                
                    
            #print(root)
            #pass