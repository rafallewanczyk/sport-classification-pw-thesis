from dataset.svw import SVW
from preprocessing.video import Video
from pathlib import Path
from tqdm import tqdm

svw = SVW()
svw.generate_features()
# seq = vid.build_sequences(False)
# print('hello world')