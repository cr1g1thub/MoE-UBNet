import os
import numpy as np
from tensorflow.keras import callbacks, optimizers
from config import CFG
from model import MOE


def load_data():
    if CFG["DATA_DIR"] is None:
        print("CFG['DATA_DIR'] is None: using random tensors (smoke test only).")
        rng = np.random.default_rng(0)
        shape = lambda m: (m, CFG["WIN"], 1)
        return (rng.standard_normal(shape(32)).astype("float32"),
                rng.standard_normal(shape(32)).astype("float32"),
                rng.standard_normal(shape(8)).astype("float32"),
                rng.standard_normal(shape(8)).astype("float32"))

    arrays = [np.load(os.path.join(CFG["DATA_DIR"], f"{name}.npy")).astype("float32")
              for name in ("X_train", "y_train", "X_val", "y_val")]
    for a in arrays:
        assert a.ndim == 3 and a.shape[1] == CFG["WIN"] and a.shape[2] == 1, \
            f"expected shape (M, {CFG['WIN']}, 1), got {a.shape}"
    return arrays


def configure_training(model, X, y, X_val, y_val):
    callback_list = [
        callbacks.ReduceLROnPlateau(factor=CFG["LR_FACTOR"], patience=CFG["LR_PATIENCE"]),
    ]

    model.gating_network.trainable = False
    for expert in model.experts:
        expert.trainable = True
    model.compile(optimizer=optimizers.Adam(learning_rate=CFG["LR"]), loss='mse')
    print("Phase 1: training experts (gate frozen)")
    model.fit(X, y, validation_data=(X_val, y_val),
              epochs=CFG["EPOCHS_P1"], batch_size=CFG["BATCH_SIZE"], callbacks=callback_list)

    model.gating_network.trainable = True
    for expert in model.experts:
        expert.trainable = True
    model.compile(optimizer=optimizers.Adam(learning_rate=CFG["LR"]), loss='mse')
    print("\nPhase 2: joint training (gate + experts)")
    model.fit(X, y, validation_data=(X_val, y_val),
              epochs=CFG["EPOCHS_P2"], batch_size=CFG["BATCH_SIZE"], callbacks=callback_list)


def main():
    X_train, y_train, X_val, y_val = load_data()
    print(f"train {X_train.shape}, val {X_val.shape}")

    model = MOE(num_experts=CFG["NUM_EXPERTS"])
    model(X_train[:1])
    for i, expert in enumerate(model.experts):
        print(f"  expert {i + 1}: {expert.count_params():,} parameters")
    print(f"  gate:     {model.gating_network.count_params():,} parameters")
    print(f"  total:    {model.count_params():,} parameters")
    if CFG["ENTROPY_LAMBDA"] > 0:
        print(f"gate entropy regularisation ON: lambda={CFG['ENTROPY_LAMBDA']}, mode={CFG['ENTROPY_MODE']}")
    else:
        print("gate entropy regularisation OFF")

    configure_training(model, X_train, y_train, X_val, y_val)

    os.makedirs(CFG["SAVE_DIR"], exist_ok=True)
    path = os.path.join(CFG["SAVE_DIR"], "moe_ubnet.weights.h5")
    model.save_weights(path)
    print(f"Saved weights to {path}")


if __name__ == "__main__":
    main()