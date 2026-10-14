import argparse
import json
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
from src.data import get_splits, make_dataset

def build_model(model_type, num_classes):
    inputs = tf.keras.Input(shape=(224, 224, 3), name="image")
    if model_type == "baseline":
        x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
        x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        return tf.keras.Model(inputs, outputs, name="cub_cnn_baseline")

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    backbone.trainable = False
    x = inputs
    if model_type == "finetune":
        # Extra augmentation; these layers are only active during training.
        x = tf.keras.layers.RandomZoom(0.15)(x)
        x = tf.keras.layers.RandomRotation(0.05)(x)
        x = tf.keras.layers.RandomTranslation(0.1, 0.1)(x)
        x = tf.keras.layers.RandomContrast(0.15)(x)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name=f"cub_mobilenetv2_{model_type}")
    return model

def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy")]
    )

def make_callbacks(model_type, append=False, best_so_far=None):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            f"models/{model_type}.keras", monitor="val_accuracy",
            save_best_only=True, mode="max", initial_value_threshold=best_so_far
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True, mode="max"
        ),
        tf.keras.callbacks.CSVLogger(f"outputs/{model_type}_training.csv", append=append)
    ]

def unfreeze_top_layers(model, count):
    backbone = next(layer for layer in model.layers if isinstance(layer, tf.keras.Model))
    backbone.trainable = True
    for layer in backbone.layers[:-count]:
        layer.trainable = False
    # Keep BatchNorm layers frozen so the pretrained statistics are not disturbed.
    for layer in backbone.layers[-count:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

def main():
    parser = argparse.ArgumentParser(description="Train a CUB-200-2011 bird classifier.")
    parser.add_argument("--dataset", default="data/CUB_200_2011")
    parser.add_argument("--model", choices=["baseline", "transfer", "finetune"], default="transfer")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=10,
                        help="Second-stage epochs with the top backbone layers unfrozen (finetune only).")
    parser.add_argument("--unfreeze-layers", type=int, default=40)
    parser.add_argument("--finetune-lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    train_df, val_df, test_df, class_names = get_splits(args.dataset, seed=args.seed)
    print(f"Images — train: {len(train_df)}, validation: {len(val_df)}, test: {len(test_df)}")
    print(f"Classes: {len(class_names)}")

    train_ds = make_dataset(train_df, args.batch_size, training=True)
    val_ds = make_dataset(val_df, args.batch_size)
    model = build_model(args.model, len(class_names))
    compile_model(model, 1e-4 if args.model == "transfer" else 1e-3)
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=make_callbacks(args.model)
    ).history
    if args.model == "finetune" and args.finetune_epochs > 0:
        # Stage 2: the head has converged, so adapt the top backbone layers at a low learning rate.
        unfreeze_top_layers(model, args.unfreeze_layers)
        compile_model(model, args.finetune_lr)
        stage2 = model.fit(
            train_ds, validation_data=val_ds, epochs=args.finetune_epochs,
            callbacks=make_callbacks(args.model, append=True, best_so_far=max(history["val_accuracy"]))
        ).history
        history = {key: history[key] + stage2[key] for key in history}
    model.save(f"models/{args.model}_final.keras")
    with open("models/class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    plt.figure()
    plt.plot(history["accuracy"], label="training accuracy")
    plt.plot(history["val_accuracy"], label="validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{args.model.title()} accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/{args.model}_accuracy.png", dpi=160)
    plt.close()
    print(f"Best checkpoint: models/{args.model}.keras")
    print("Class names saved to models/class_names.json")

if __name__ == "__main__":
    main()
