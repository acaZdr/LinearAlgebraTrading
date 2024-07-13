import os
import zipfile

# Specify the directory you want to start from
rootDir = '/Users/aleksandarzdravkovic/PycharmProjects/LinearAlgebraTrading/data'

for dirName, subdirList, fileList in os.walk(rootDir):
    print(f'Found directory: {dirName}')
    for fname in fileList:
        if fname.endswith('.zip'):
            print('\tFound file: %s' % fname)
            # Check if the extracted folder already exists
            extracted_folder = os.path.join(dirName, os.path.splitext(fname)[0])
            if not os.path.exists(extracted_folder):
                try:
                    with zipfile.ZipFile(os.path.join(dirName, fname), 'r') as zip_ref:
                        zip_ref.extractall(dirName)
                        print(f'\tUnzipped file: {fname}')
                except zipfile.BadZipFile:
                    print(f'\tFile {fname} is not a valid zip file.')
                    continue
            else:
                print(f'\tFile {fname} already extracted.')