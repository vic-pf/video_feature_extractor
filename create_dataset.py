import torch
import os
import numpy as np
import pickle
import re
from gensim.models.keyedvectors import KeyedVectors

_CLASS_ID = {'fast-food': '0', 'food-drink': '1', 'non-food': '2', 'supermarket': '3'}

def load_pretrained_embeddings():
    print("Loading pre-trained Word2Vec model...")
    model = api.load("word2vec-google-news-300")
    return model

def _tokenize_text(sentence):
    w = re.findall(r"[\w']+", str(sentence))
    return w

def create_text_features(words, max_words, we, we_dim):
    raw_text = ' '.join(words)
    words = [word for word in words if word in we.vocab]
    text = np.zeros((max_words, we_dim), dtype=np.float32)
    text_mask = np.zeros(max_words, dtype=np.float32)
    nwords = min(len(words), max_words)
    if nwords > 0:
        text[:nwords] = we[words][:nwords]
        text_mask[:nwords] = 1
    text = torch.from_numpy(text).float()
    text_mask = torch.from_numpy(text_mask).float()

    return text, text_mask, raw_text

def load_features(feature_dir):
    """
    Load video and audio features from .npy files and save them into a pickle file.
    """
    for _TYPE in os.listdir(feature_dir):
        if _TYPE.lower() not in ('train', 'test', 'val'):
            continue

        type_path = os.path.join(feature_dir, _TYPE)

        # List to store the features
        feature_list = []

        # Iterate over all classes
        for class_name in os.listdir(type_path):
            if class_name not in _CLASS_ID:
                continue

            class_path = os.path.join(type_path, class_name)

            if not os.path.isdir(class_path):
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

                add = False
                # Load 2D, 3D, and audio features
                try:
                    feature_2d_path = os.path.join(video_path, '2d.npy')
                    feature_3d_path = os.path.join(video_path, '3d.npy')
                    audio_feature_path = os.path.join(video_path, 'audio.npy')

                    if os.path.exists(feature_2d_path):
                        video_data['2d'] = np.load(feature_2d_path)
                        add = not add

                    if os.path.exists(feature_3d_path):
                        video_data['3d'] = np.load(feature_3d_path)
                        add = not add

                    if os.path.exists(audio_feature_path):
                        video_data['audio'] = np.load(audio_feature_path)
                        add = not add

                    # Append to the list
                    if add:
                        feature_list.append(video_data)

                except Exception as e:
                    print(f"Error loading features for {video_id}: {e}")

        # Save the feature list to a pickle file
        pickle_file_path = os.path.join(feature_dir, f'{_TYPE}.pkl')
        with open(pickle_file_path, 'wb') as f:
            pickle.dump(feature_list, f)

        print(f"Features saved to {pickle_file_path}")
        print(f"Dataset size {len(feature_list)}")

# Set the feature directory path
_FEATURE_DIR_ROOT = '/dataset/youtube/videos/features/'
we = load_pretrained_embeddings()

# Call the function to load features and save them to a pickle file
load_features(_FEATURE_DIR_ROOT)
