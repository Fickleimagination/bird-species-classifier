from pathlib import Path
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

IMAGE_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE

def read_cub_metadata(dataset_dir):
    root = Path(dataset_dir)
    required = ["images.txt", "image_class_labels.txt", "train_test_split.txt", "classes.txt"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing CUB metadata files in {root}: {missing}. "
            "Check that CUB_200_2011 was extracted correctly."
        )

    images = pd.read_csv(root / "images.txt", sep=r"\s+", names=["image_id", "relative_path"])
    labels = pd.read_csv(root / "image_class_labels.txt", sep=r"\s+", names=["image_id", "class_id"])
    split = pd.read_csv(root / "train_test_split.txt", sep=r"\s+", names=["image_id", "is_train"])
    classes = pd.read_csv(root / "classes.txt", sep=r"\s+", names=["class_id", "class_name"])

    df = images.merge(labels, on="image_id").merge(split, on="image_id")
    df = df.merge(classes, on="class_id")
    df["path"] = df["relative_path"].map(lambda p: str(root / "images" / p))
    df["label"] = df["class_id"] - 1
    absent = [p for p in df["path"] if not Path(p).is_file()]
    if absent:
        raise FileNotFoundError(f"Could not find image file, for example: {absent[0]}")
    class_names = classes.sort_values("class_id")["class_name"].tolist()
    return df, class_names

def _load_image(path, label, training=False):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    if training:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=12.0)
        image = tf.clip_by_value(image, 0.0, 255.0)
    return image, label

def make_dataset(frame, batch_size=32, training=False):
    paths = frame["path"].astype(str).to_numpy()
    labels = frame["label"].astype("int32").to_numpy()
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(len(frame), 5000), reshuffle_each_iteration=True)
    ds = ds.map(
        lambda p, y: _load_image(p, y, training),
        num_parallel_calls=AUTOTUNE
    )
    return ds.batch(batch_size).prefetch(AUTOTUNE)

def get_splits(dataset_dir, validation_fraction=0.15, seed=42):
    df, class_names = read_cub_metadata(dataset_dir)
    official_train = df[df["is_train"] == 1].copy()
    test = df[df["is_train"] == 0].copy()
    train, validation = train_test_split(
        official_train,
        test_size=validation_fraction,
        random_state=seed,
        stratify=official_train["label"]
    )
    return train, validation, test, class_names
