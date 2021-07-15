#!/usr/bin/env python3
from os import listdir
import numpy as np
import pandas as pd
from AudioFeatureExtractor import AudioFeatureExtractor

genres = ['Blues', 'Classical', 'Country', 'Disco', 'Hiphop', 'Jazz', 'Metal', 'Pop', 'Rock']
csv_name = 'test_feature.csv' # TODO - Update when have real csv name

# Create the AudioFeatureExtractor Object
audio_feature_extractor = AudioFeatureExtractor()

# Get the column names for the data
feature_list = audio_feature_extractor.generate_feature_list()

all_feature_data = []

# Cycle through each Genre Folder
for genre in genres:
    # Extract features from each audio file in the genre folder
    folder_path = None # TODO - Generate from the folder path
    audio_files = listdir(folder_path)
    feature_data_for_this_genre = np.array([audio_feature_extractor.extract((folder_path + '/' + audio_file), genre) for audio_file in audio_files])
    all_feature_data.append(feature_data_for_this_genre)

# Save the data to a CSV
all_feature_data = np.vstack(all_feature_data)
all_feature_df = pd.DataFrame(data=all_feature_data, columns=feature_list)
all_feature_df.to_csv(csv_name, sep=',', index=False)