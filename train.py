"""Train the mesh GNN and the node-wise MLP baseline: python train.py  (~25 min on one CPU core)"""
import json
import os
import pickle
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

from config import ENGINEERED_FEATURES, BATCH_GRAPHS, EPOCHS, HIDDEN, LEARNING_RATE, MESSAGE_STEPS, MODEL_DIR, RESULTS_DIR, SEED
from src.dataset import load
from src.gnn import MeshGNN, Normalizer, train
from src.simulate import to_graph

MODELS = {"gnn": True, "mlp": False}


def main():
    tf.random.set_seed(SEED)
    splits = load()
    graphs = {k: [to_graph(d["design"], d["result"], ENGINEERED_FEATURES) for d in v] for k, v in splits.items()}
    norm = Normalizer().fit(graphs["train"])
    train_graphs = [norm(g) for g in graphs["train"]]
    val_graphs = [norm(g) for g in graphs["val"]]
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / "normalizer.pkl", "wb") as f:
        pickle.dump(norm, f)
    histories = {}
    for name, messages in MODELS.items():
        start = time.time()
        model = MeshGNN(HIDDEN, MESSAGE_STEPS, messages)
        histories[name] = train(model, train_graphs, val_graphs, EPOCHS, BATCH_GRAPHS, LEARNING_RATE, SEED,
                                log=lambda s: print(f"[{name}] {s}", flush=True))
        model.save_weights(MODEL_DIR / f"{name}.weights.h5")
        print(f"[{name}] trained in {(time.time() - start) / 60:.1f} min", flush=True)
    (RESULTS_DIR / "training_history.json").write_text(json.dumps(histories))


if __name__ == "__main__":
    main()
