import torch
import os
import numpy as np
import pickle
import re
from gensim.models.keyedvectors import KeyedVectors

_CLASS_ID = {'fast-food': '0', 'food-drink': '1', 'non-food': '2', 'supermarket': '3'}

# Set the feature directory path
_FEATURE_DIR_ROOT = '/home/victoria/dataset/youtube/videos/features'
we = KeyedVectors.load_word2vec_format('/home/victoria/dataset/GoogleNews-vectors-negative300.bin', binary=True)

def _tokenize_text(sentence):
    w = re.findall(r"[\w']+", str(sentence))
    return w

def create_text_features(words, max_words, we, we_dim):
    raw_text = ' '.join(words)
    # Update the code to use key_to_index instead of vocab
    words = [word for word in words if word in we.key_to_index]
    text = np.zeros((max_words, we_dim), dtype=np.float32)
    text_mask = np.zeros(max_words, dtype=np.float32)
    nwords = min(len(words), max_words)
    if nwords > 0:
        text[:nwords] = np.array([we[word] for word in words[:nwords]])
        text_mask[:nwords] = 1
    text = torch.from_numpy(text).float()
    text_mask = torch.from_numpy(text_mask).float()

    return text, text_mask, raw_text

def load_features(feature_dir):
    """
    Load video and audio features from .npy files and save them into a pickle file.
    """
    print(f"Starting to load features from {feature_dir}...")
    
    for _TYPE in os.listdir(feature_dir):
        if _TYPE.lower() not in ('train', 'test', 'val'):
            print(f"Skipping unknown type {_TYPE}")
            continue

        type_path = os.path.join(feature_dir, _TYPE)
        print(f"Processing {_TYPE} data in {type_path}...")

        # List to store the features
        feature_list = []

        # Iterate over all classes
        for class_name in os.listdir(type_path):
            if class_name not in _CLASS_ID:
                print(f"Skipping unknown class {class_name}")
                continue

            class_path = os.path.join(type_path, class_name)
            print(f"Processing class {class_name} in {class_path}...")

            if not os.path.isdir(class_path):
                print(f"Skipping {class_path} as it is not a directory")
                continue

            # Iterate over all video files in the class directory
            for video_id in os.listdir(class_path):
                video_data = {'id': video_id, 'class': class_name, 'label': _CLASS_ID[class_name]}

                # Define the caption
                if class_name == 'non-food':
                    caption = 'other'
                elif class_name == 'food-drink':
                    caption = 'aliment-drink'
                else:
                    caption = class_name

                words = _tokenize_text(caption)
                text, text_mask, raw_text = create_text_features(words, 20, we, 300)

                video_data['caption'] = caption
                video_data['text'] = text
                video_data['text_mask'] = text_mask
                video_data['raw_text'] = raw_text

                video_path = os.path.join(class_path, video_id)
                print(f"Processing video {video_id} at {video_path}...")

                add = False
                # Load 2D, 3D, and audio features
                try:
                    feature_2d_path = os.path.join(video_path, '2d.npy')
                    feature_3d_path = os.path.join(video_path, '3d.npy')
                    audio_feature_path = os.path.join(video_path, 'audio.npy')

                    if os.path.exists(feature_2d_path):
                        print(f"Loading 2D features from {feature_2d_path}")
                        video_data['2d'] = np.load(feature_2d_path)
                        add = True

                    if os.path.exists(feature_3d_path):
                        print(f"Loading 3D features from {feature_3d_path}")
                        video_data['3d'] = np.load(feature_3d_path)
                        add = True

                    if os.path.exists(audio_feature_path):
                        print(f"Loading audio features from {audio_feature_path}")
                        video_data['audio'] = np.load(audio_feature_path)
                        add = True

                    # Append to the list
                    if add:
                        feature_list.append(video_data)
                        print(f"Features loaded and added for video {video_id}")
                    else:
                        print(f"No features found for video {video_id}")

                except Exception as e:
                    print(f"Error loading features for {video_id}: {e}")

        # Save the feature list to a pickle file
        pickle_file_path = os.path.join(feature_dir, f'{_TYPE}.pkl')
        with open(pickle_file_path, 'wb') as f:
            pickle.dump(feature_list, f)

        print(f"Features saved to {pickle_file_path} for {_TYPE} data")
        print(f"Dataset size for {_TYPE}: {len(feature_list)} samples")

# Call the function to load features and save them to a pickle file
load_features(_FEATURE_DIR_ROOT)
