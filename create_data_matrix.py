import numpy as np
import matplotlib.pyplot as plt
import sklearn

class Song_Features:
    """
    A container for a musical excerpt's features
    """
    def __init__(self, name, label):
        """
        Constructs a new song features object
        :param name: Name of song
        :param label: True for horror piece, False otherwise
        """
        self.name = name
        self.label = label
        self.rms = None
        self.spec_cent= None
        self.spec_bw = None
        self.spec_rf = None
        self.zcr = None
        self.mfcc = None
        self.chroma_stft = None

    def to_csv_string(self):  # TODO add sd?
        """
        :return: A string containing relevant features. The string is ready to be written to a CSV file.
        """
        csv_string = f'{self.name} {np.mean(self.chroma_stft)} {np.mean(self.rms)} {np.mean(self.spec_cent)}'
        csv_string += f' {np.mean(self.spec_bw)} {np.mean(self.spec_rf)} {np.mean(self.zcr)}'
        for frame in self.mfcc:
            csv_string += f' {np.mean(frame)}'
        csv_string += int(self.label)
        return csv_string

def get_features(x, sr, name, label):
    """
    Gets features from a particular track.
    :param x: librosa data for the track
    :param sr: track's sample rate
    :param name: name of the track
    :param label: True for a horror piece, False otherwise
    :return: A class with all of the provided data's features inside it
    """
    features = Song_Features(name, label)
    features.rms = librosa.feature.rms(x)[0]
    features.spec_cent = librosa.feature.spectral_centroid(x, sr=sr)
    features.spec_bw = librosa.feature.spectral_bandwidth(x, sr=sr)
    features.spec_rf= librosa.feature.spectral_rolloff(x, sr=sr)
    features.zcr = librosa.feature.zero_crossing_rate(x, sr=sr)
    features.mfcc = librosa.feature.mfcc(x, sr=sr)
    features.chroma_stft = librosa.feature.chroma_stft(x, sr=sr)
    return features

def get_dataset(horror_path, non_horror_path, save_name):
    """
    Extracts features puts them in a CSV file. Each row in the file contains the name of the track,
    the various features and the track's label (1 for horror, 0 otherwise)
    :param horror_path: Path to the folder containing the horror tracks
    :param non_horror_path: Path to the folder containing the non horror tracks
    :param save_name: Name of the CSV file to save into
    """
    with open(save_name + '.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)


path = 'C:\\Users\\barkr\\Music\\Joanna Newsom - Have One On Me [2010-MP3-Cov][Bubanee]\\CD 2\\07 On A Good ' \
    'Day.mp3'
x1, sr1 = librosa.load(path)
get_features(x1, sr1)