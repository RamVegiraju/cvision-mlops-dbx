"""Constants shared by every stage of the pipeline."""

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
IMAGENETTE_DIRNAME = "imagenette2-160"

# Imagenette = 10 easily-separable ImageNet classes, keyed by WordNet synset id (the folder name).
SYNSET_TO_LABEL = {
    "n01440764": "tench",
    "n02102040": "english_springer",
    "n02979186": "cassette_player",
    "n03000684": "chain_saw",
    "n03028079": "church",
    "n03394916": "french_horn",
    "n03417042": "garbage_truck",
    "n03425413": "gas_pump",
    "n03445777": "golf_ball",
    "n03888257": "parachute",
}
CLASS_NAMES = sorted(SYNSET_TO_LABEL.values())
LABEL_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# ImageNet preprocessing — must match the pretrained backbones.
IMAGE_SIZE = 224
RESIZE_SIZE = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SPLITS = ("train", "val", "test")
VAL_FRACTION = 0.1  # carved deterministically out of Imagenette's train folder; its val folder becomes test

MANIFEST_FILENAME = "manifest.parquet"
