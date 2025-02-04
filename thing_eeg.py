import os
import numpy as np
from PIL import Image


eeg_data_train = np.load(os.path.join('/root/autodl-tmp/DreamDiffusion/sub-01/preprocessed_eeg_training.npy'), allow_pickle=True).item()
reshaped_data = eeg_data_train[:, :2, :, :]
np.save('output_file.npy', reshaped_data)

eeg_data_test = np.load(os.path.join('/root/autodl-tmp/DreamDiffusion/sub-01/preprocessed_eeg_test.npy'), allow_pickle=True).item()
img_metadata = np.load(os.path.join('/root/autodl-tmp/DreamDiffusion/image_metadata.npy'), allow_pickle=True).item()
n_train_img = len(img_metadata['train_img_concepts'])
n_train_concepts = len(np.unique(img_metadata['train_img_concepts']))
n_train_img_per_concept = int(n_train_img / n_train_concepts)

train_img_idx =  0
