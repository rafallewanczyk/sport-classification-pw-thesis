from dataset.svw import SVW
from preprocessing.dto.sequence_dto import SequenceDto
from pathlib import Path
import numpy as np

from preprocessing.video import Video
from preprocessing.inference_service.external_model.mobilenet.mobilenet_service import MobilenetService

vid = Video(Path("/home/rafa/SVW/Videos/hammerthrow/9720___7810ba3976b54da993f07f56693b0501.mp4"), allow_processing_pipeline=True)

mobilenet_service = MobilenetService()
svw = SVW()
svw.generate_features()
# svw.preprocess_videos(mobilenet_service)
# seq = vid.build_sequences(use_dump=False)
# x = vid.extend_detections(mobilenet_service)
# print("hello")