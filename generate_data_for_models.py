from dataset.svw import SVW
from model.train_notebooks._notebook_utils import DEFAULT_CLASSES, print_shapes, plot_histogram

from model.abc_model import ABCModel
from model.frame import Frame  # noqa
from model.skeletons import Skeletons  # noqa
from model.concated import Concated  # noqa
from model.concated_frame import ConcatedFrame  # noqa
from model.skconcated import Skconcated  # noqa
from model.sp import Sp  # noqa
from model.sp_frame import SpFrame  # noqa
from model.sp_concated_frame import SpConcatedFrame  # noqa
from model.sk_frame import SkFrame  # noqa
from model.sk_concated_frame import SkConcatedFrame
from tqdm import tqdm
from tensorflow import keras
from time import time

import gc

svw = SVW(classes=DEFAULT_CLASSES)
ABCModel.CLASSES_NUMBER = len(svw.classes)

# models = [Skeletons, Frame, Concated, ConcatedFrame, Skconcated, Sp, SpFrame, SpConcatedFrame, SkFrame, SkConcatedFrame]
models = [ConcatedFrame]
for model in models:
    print(model.__name__)
    for split in range(1, 4):
        _model = model()

        x_train, y_train, x_test, y_test, x_val, y_val,  _, videos_test, _, histogram_train, histogram_test, _ = svw.get_as_x_y(
            max_person_number=model.PERSON_LIMIT,
            feature_types=model.get_required_features_names(),
            use_full_dump=False, use_videos_dump=(split in [2, 3]), split_id=split
        )
        start = time()
        history = _model.train(50, x_train, y_train, x_val, y_val)
        print(f"training took {time()-start}")
        acc, y_pred, y_true = _model.evaluate_weighted_avg(x_test, videos_test, svw.translate_class)
        print(f"model {model.__name__}, split {split}, acc {acc}")

        del x_train, y_train, x_test, y_test, _, videos_test, histogram_train, histogram_test
        keras.backend.clear_session()
        gc.collect()
