import os
import shutil
import numpy as np

execution_path = os.getcwd()
base_dir = os.path.join(execution_path, 'image-dataset')

CLS_1 = 'horse'
CLS_2 = 'lion'
TRAIN_RATIO = 0.7
VALID_RATIO = 0.2
DATA_DIR = r'./images'

raw_no_of_files = {}
classes = [CLS_1, CLS_2]

number_of_samples = [(dir, len(os.listdir(os.path.join(base_dir, dir)))) for dir in classes]
print(number_of_samples)

if not os.path.exists(DATA_DIR): os.mkdir(DATA_DIR)

# Katalogi do zbiorów: train, valid, test
train_dir = os.path.join(DATA_DIR, 'train')
valid_dir = os.path.join(DATA_DIR, 'valid')
test_dir = os.path.join(DATA_DIR, 'test')

train_cls_1_dir = os.path.join(train_dir, CLS_1)
valid_cls_1_dir = os.path.join(valid_dir, CLS_1)
test_cls_1_dir = os.path.join(test_dir, CLS_1)

train_cls_2_dir = os.path.join(train_dir, CLS_2)
valid_cls_2_dir = os.path.join(valid_dir, CLS_2)
test_cls_2_dir = os.path.join(test_dir, CLS_2)

for dir in (train_dir, valid_dir, test_dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

for dir in (train_cls_1_dir, valid_cls_1_dir, test_cls_1_dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

for dir in (train_cls_2_dir, valid_cls_2_dir, test_cls_2_dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

print('[INFO] Wczytanie nazw plików...')
cls_1_names = os.listdir(os.path.join(base_dir, CLS_1))
cls_2_names = os.listdir(os.path.join(base_dir, CLS_2))

print('[INFO] Walidacja poprawności nazw...')
cls_1_names = [fname for fname in cls_1_names if fname.split('.')[1].lower() in ['jpg', 'png', 'jpeg']]
cls_2_names = [fname for fname in cls_2_names if fname.split('.')[1].lower() in ['jpg', 'png', 'jpeg']]

# Przetasowanie nazw plików
np.random.shuffle(cls_1_names)
np.random.shuffle(cls_2_names)

print(f'[INFO] Liczba obrazów w zbiorze {CLS_1}: {len(cls_1_names)}')
print(f'[INFO] Liczba obrazów w zbiorze {CLS_2}: {len(cls_2_names)}')

train_idx_cls_1 = int(TRAIN_RATIO * len(cls_1_names))
valid_idx_cls_1 = train_idx_cls_1 + int(VALID_RATIO * len(cls_1_names))

train_idx_cls_2 = int(TRAIN_RATIO * len(cls_2_names))
valid_idx_cls_2 = train_idx_cls_2 + int(VALID_RATIO * len(cls_2_names))

print('[INFO] Kopiowanie plików do katalogów docelowych...')
for i, fname in enumerate(cls_1_names):
    if i <= train_idx_cls_1:
        src = os.path.join(base_dir, CLS_1, fname)
        dst = os.path.join(train_cls_1_dir, fname)
        shutil.copyfile(src, dst)
    if train_idx_cls_1 < i <= valid_idx_cls_1:
        src = os.path.join(base_dir, CLS_1, fname)
        dst = os.path.join(valid_cls_1_dir, fname)
        shutil.copyfile(src, dst)
    if valid_idx_cls_1 < i <= len(cls_1_names):
        src = os.path.join(base_dir, CLS_1, fname)
        dst = os.path.join(test_cls_1_dir, fname)
        shutil.copyfile(src, dst)

for i, fname in enumerate(cls_2_names):
    if i <= train_idx_cls_2:
        src = os.path.join(base_dir, CLS_2, fname)
        dst = os.path.join(train_cls_2_dir, fname)
        shutil.copyfile(src, dst)
    if train_idx_cls_2 < i <= valid_idx_cls_2:
        src = os.path.join(base_dir, CLS_2, fname)
        dst = os.path.join(valid_cls_2_dir, fname)
        shutil.copyfile(src, dst)
    if valid_idx_cls_2 < i <= len(cls_2_names):
        src = os.path.join(base_dir, CLS_2, fname)
        dst = os.path.join(test_cls_2_dir, fname)
        shutil.copyfile(src, dst)

print(f'[INFO] Liczba obrazów klasy {CLS_1} w zbiorze treningowym: {len(os.listdir(train_cls_1_dir))}')
print(f'[INFO] Liczba obrazów klasy {CLS_1} w zbiorze validacyjnym: {len(os.listdir(valid_cls_1_dir))}')
print(f'[INFO] Liczba obrazów klasy {CLS_1} w zbiorze testowym: {len(os.listdir(test_cls_1_dir))}')
print(f'[INFO] Liczba obrazów klasy {CLS_2} w zbiorze treningowym: {len(os.listdir(train_cls_2_dir))}')
print(f'[INFO] Liczba obrazów klasy {CLS_2} w zbiorze validacyjnym: {len(os.listdir(valid_cls_2_dir))}')
print(f'[INFO] Liczba obrazów klasy {CLS_2} w zbiorze testowym: {len(os.listdir(test_cls_2_dir))}')