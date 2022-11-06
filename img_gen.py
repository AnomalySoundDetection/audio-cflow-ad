'''
Converts waveform data into log-mel spectrogram images.
'''

import common, os, cv2

INPUT_DIR = '../dev_data/valve/test'
OUTPUT_DIR = "data/DCASE/valve/test"

# iterate over all files in the directory
for file in os.listdir(INPUT_DIR):
    print("converting", file)
    img = common.file_to_vector_array(os.path.join(INPUT_DIR, file))
    # normalize
    img += 80
    img /= 80
    # save
    cv2.imwrite(os.path.join(OUTPUT_DIR, file + '.png'), img * 255)