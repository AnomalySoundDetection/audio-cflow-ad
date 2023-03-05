'''
Converts waveform data into log-mel spectrogram images.
'''
from __future__ import print_function
import argparse

print("Importing libraries...")

import common, os, cv2
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Image Generator')
parser.add_argument('--machine-type', default='fan', type=str, metavar='<machine_name>',
                        help='machine-type: ToyCar/ToyConveyor/pump/slider/valve/fan')
args = parser.parse_args()
machine_type = args.machine_type

for path in ["ToyCar", "ToyConveyor", "pump", "slider", "valve", "fan"]:
    if path != machine_type:
        continue

    print("Processing", path)

    INPUT_DIR = '/mnt/Directory/dev_data/' + path + "/"
    OUTPUT_DIR = "data/" + path + "/"

    for phase in ['train', 'test']:
        try:
            os.makedirs(OUTPUT_DIR + phase)
            print("Created " + OUTPUT_DIR + phase)
        except:
            print("Path already exists, continuing...")
        # iterate over all files in the directory
        for file in tqdm(os.listdir(INPUT_DIR + phase)):
            #print("converting", file)
            '''img = common.file_to_vector_array(os.path.join(INPUT_DIR + phase, file))
            # normalize
            img += 80
            img /= 80
            #img -= img.min()
            #img /= img.max()
            # save
            '''
            #print("converting", file)
            img = common.file_to_vector_array(os.path.join(INPUT_DIR + phase, file), n_mels=256)
            # normalize
            #print(img.min(), img.max())
            img -= img.min() / 1.2
            img /= img.max() / 1.2
            img[img > 1] = 1
            img[img < 0] = 0
            img *= 255
            cv2.imwrite(os.path.join(OUTPUT_DIR + phase, file.split(".")[0] + '.png'), img)
