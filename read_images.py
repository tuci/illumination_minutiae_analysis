import glob, os, cv2, re
import numpy as np

class ImageSet:
    def __init__(self, image, mask, annotation, name, t, i, d):
        self.image = image.astype(np.uint8)
        self.mask = mask.astype(bool)
        self.annotated = annotation
        self.type=t
        self.direction = d
        self.img_name = name
        self.i = i 

def read_images(folder):
    files = glob.glob(f'{folder}/*.jpg')
    files = sorted([f for f in files if '_annotated' not in f])

    # group by acquisition
    acquisitions = []
    for file in files:
        acq_id = int(file.split("_")[-1].split(".")[0])

        while len(acquisitions) < acq_id:
            acquisitions.append([])

        acquisitions[acq_id - 1].append(file)

    imageset = []

    for i, acquisition in enumerate(acquisitions):
        imageset.append([])
        for imgFile in acquisition:
            file_name = imgFile.split('.')[-2]
            name_split = re.split('[/_]', file_name)
            light_direction = name_split[-3]
            mark_type = name_split[-2]
            img_id = name_split[-1]
            print(f'Reading.... {file_name}')

            img = cv2.imread(f'{imgFile}')
            annotation = cv2.imread(f'{folder}/{light_direction}_{mark_type}_{img_id}_annotated.jpg')
            mask = cv2.imread(f'{folder}/{light_direction}_{mark_type}_{img_id}_mask.png', 0)
            mark = ImageSet(img, mask, annotation, name=file_name, t=mark_type, 
                            d=light_direction, i=img_id)    
            imageset[i].append(mark)   
        
    return imageset


