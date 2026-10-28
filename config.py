CFG = {
    "DATA_DIR": None,
    "SAVE_DIR": "./outputs",
    "WIN": 5000, # window length T in samples (5 s at 1000 Hz for the Kansas dataset)

    "NUM_EXPERTS": 3,
    "KERNEL_SIZE": 8,
    "EXPERTS_PER_WIDTH": 2,
    "LSTM_BASE": 128,
    "ATTN_HEADS": 4,
    "ATTN_KEY_DIM": 128,

    "LR": 1e-3,
    "BATCH_SIZE": 32,
    "EPOCHS_P1": 40, # phase 1: experts only, gate frozen
    "EPOCHS_P2": 60,  # phase 2: gate + experts jointly
    "LR_FACTOR": 0.5,
    "LR_PATIENCE": 10,

    "ENTROPY_LAMBDA": 0.1,
    "ENTROPY_MODE": "per_sample",
}