import sys
import cv2
import os
import glob
import numpy as np

def update_progress(current, total):
    print('Info: Processed [{0}/{1}]'.format(current, total))
    if (current == total):
        print('Info: Processing done!')

def write_to_file(name, dir, data):
    file = open(dir + '/' + name + '.txt', 'w')
    temp_data = str(data).strip('[]').replace('.', '')
    clean_data = " ".join(temp_data.split())
    file.write(clean_data)
    file.close()

def extract_features(img):
    sift = cv2.SIFT()
    kps, des = sift.detectAndCompute(img, None)
    return des.flatten();

def convert_to_grayscale(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def get_filenames(dir):
    files = []
    files.extend(glob.glob1(dir, '*.jpg'))
    files.extend(glob.glob1(dir, '*.jpeg'))
    files.extend(glob.glob1(dir, '*.png'))
    return files

def main(argv):
    argc = len(argv)

    np.set_printoptions(threshold=np.nan)

    if (argc > 1):
        dir = argv[1]
        files = get_filenames(dir);
        index = 0
        for file in files:
            img = convert_to_grayscale(dir + '/' + file)
            features = extract_features(img)
            write_to_file(os.path.splitext(file)[0], dir, features)
            index = index + 1
            update_progress(index, len(files))

        if (len(files) == 0):
            print('Warning: Could not find images of type png, jpg, jpeg in directory!')
    else:
        print('Warning: No directory specified!')


if __name__ == "__main__":
    main(sys.argv)