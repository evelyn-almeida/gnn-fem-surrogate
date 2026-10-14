"""Message-passing graph network on FE meshes (MeshGraphNets-style), implemented from scratch in TensorFlow."""
import numpy as np
import tensorflow as tf


def _mlp(hidden, out, norm=True):
    layers = [tf.keras.layers.Dense(hidden, activation="relu"), tf.keras.layers.Dense(out)]
    if norm:
        layers.append(tf.keras.layers.LayerNormalization())
    return tf.keras.Sequential(layers)


class MeshGNN(tf.keras.Model):
    """Encode → process (message passing along mesh edges) → decode a stress value per node.

    messages=False removes the edge updates: every node is then predicted from its own features only,
    which gives a node-wise MLP baseline of the same depth and width.
    """

    def __init__(self, hidden=64, steps=10, messages=True):
        super().__init__()
        self.messages = messages
        self.node_encoder, self.edge_encoder = _mlp(hidden, hidden), _mlp(hidden, hidden)
        self.edge_blocks = [_mlp(hidden, hidden) for _ in range(steps)] if messages else []
        self.node_blocks = [_mlp(hidden, hidden) for _ in range(steps)]
        self.decoder = _mlp(hidden, 1, norm=False)

    def call(self, x, e, senders, receivers):
        h = self.node_encoder(x)
        if self.messages:
            m = self.edge_encoder(e)
        for i, node_block in enumerate(self.node_blocks):
            if self.messages:
                m = m + self.edge_blocks[i](tf.concat([m, tf.gather(h, senders), tf.gather(h, receivers)], -1))
                incoming = tf.math.unsorted_segment_mean(m, receivers, tf.shape(h)[0])
            else:
                incoming = tf.zeros_like(h)
            h = h + node_block(tf.concat([h, incoming], -1))
        return self.decoder(h)[:, 0]


class Normalizer:
    def fit(self, graphs):
        x = np.concatenate([g["x"] for g in graphs])
        e = np.concatenate([g["e"] for g in graphs])
        self.x_mean, self.x_std = x.mean(0), x.std(0) + 1e-6
        self.e_mean, self.e_std = e.mean(0), e.std(0) + 1e-6
        return self

    def __call__(self, graph):
        return {**graph, "x": (graph["x"] - self.x_mean) / self.x_std, "e": (graph["e"] - self.e_mean) / self.e_std}


def batch(graphs):
    """Disjoint union of graphs: one big graph with shifted edge indices."""
    offsets = np.cumsum([0] + [len(g["x"]) for g in graphs[:-1]])
    return (np.concatenate([g["x"] for g in graphs]), np.concatenate([g["e"] for g in graphs]),
            np.concatenate([g["senders"] + o for g, o in zip(graphs, offsets)]),
            np.concatenate([g["receivers"] + o for g, o in zip(graphs, offsets)]),
            np.concatenate([g["y"] for g in graphs]))


def signature(graph):
    return [tf.TensorSpec([None, graph["x"].shape[1]], tf.float32), tf.TensorSpec([None, graph["e"].shape[1]], tf.float32),
            tf.TensorSpec([None], tf.int32), tf.TensorSpec([None], tf.int32), tf.TensorSpec([None], tf.float32)]


def train(model, train_graphs, val_graphs, epochs, batch_graphs, lr, seed, log=print):
    """Mean squared error on the residual (FEM − beam theory) / nominal stress."""
    steps_per_epoch = int(np.ceil(len(train_graphs) / batch_graphs))
    schedule = tf.keras.optimizers.schedules.CosineDecay(lr, epochs * steps_per_epoch, alpha=0.05)
    opt = tf.keras.optimizers.Adam(schedule)

    spec = signature(train_graphs[0])

    @tf.function(input_signature=spec)
    def step(x, e, s, r, y):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(model(x, e, s, r) - y))
        opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
        return loss

    @tf.function(input_signature=spec)
    def evaluate(x, e, s, r, y):
        return tf.reduce_mean(tf.square(model(x, e, s, r) - y))

    rng = np.random.default_rng(seed)
    val_batch = batch(val_graphs)
    history = []
    for epoch in range(epochs):
        order = rng.permutation(len(train_graphs))
        losses = [float(step(*batch([train_graphs[i] for i in order[k:k + batch_graphs]])))
                  for k in range(0, len(order), batch_graphs)]
        history.append({"epoch": epoch + 1, "train": float(np.mean(losses)), "val": float(evaluate(*val_batch))})
        log(f"epoch {epoch + 1:3d}  train {history[-1]['train']:.4f}  val {history[-1]['val']:.4f}")
    return history


_compiled = {}


def predict(model, graph):
    """Stress / nominal stress = beam theory + learned correction."""
    if id(model) not in _compiled:
        _compiled[id(model)] = tf.function(model.call, input_signature=signature(graph)[:4])
    x, e, s, r, _ = batch([graph])
    return graph["beam"] + _compiled[id(model)](x, e, s, r).numpy()
