import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib

matplotlib.use('TkAgg')  # or 'Qt5Agg', 'WebAgg', etc.


data = mpimg.imread("/home/liora/Lior/Datasets/record3d/iphone_with_depth/1_preprocessed/depth/100.png")
plt.imshow(data, cmap='hot', interpolation='nearest')

plt.show()

from pathlib import Path
from pprint import pformat

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_retrieval,
)



images = Path("/home/liora/Lior/Datasets/svo/global/merge/3_4/images")

outputs = Path("/home/liora/Lior/Datasets/svo/global/merge/3_4/sfm/")
sfm_pairs = outputs / "pairs-netvlad.txt"
sfm_dir = outputs / "sfm_superpoint+superglue"

retrieval_conf = extract_features.confs["netvlad"]
feature_conf = extract_features.confs["superpoint_inloc"]
matcher_conf = match_features.confs["superglue"]




retrieval_path = extract_features.main(retrieval_conf, images, outputs)
pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)




feature_path = extract_features.main(feature_conf, images, outputs)
match_path = match_features.main(
    matcher_conf, sfm_pairs, feature_conf["output"], outputs
)

model = reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path)
