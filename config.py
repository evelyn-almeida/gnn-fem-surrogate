from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models"

SEED = 42

# Material: structural steel, plane stress
YOUNG = 210e3        # MPa
POISSON = 0.30
THICKNESS = 5.0      # mm

# Bracket design space (mm, N)
LENGTH = (80.0, 140.0)
HEIGHT = (40.0, 70.0)
HOLE_RADIUS = (4.0, 11.0)
LOAD = (2_000.0, 10_000.0)
LOAD_ANGLE = (-90.0, 90.0)     # degrees: 0 = axial pull, ±90 = transverse shear
MAX_ELEMENT_AREA = 12.0        # mm², about 450 nodes per mesh

N_TRAIN, N_VAL, N_TEST, N_OOD = 400, 60, 100, 100
TRAIN_HOLES = (1, 2)           # holes per design seen in training
OOD_HOLES = 3                  # out-of-distribution test: three holes

# GNN
ENGINEERED_FEATURES = False   # True gives nodes hand-crafted hole descriptors (see simulate.to_graph)
HIDDEN = 48
MESSAGE_STEPS = 8
EPOCHS = 40
BATCH_GRAPHS = 8
LEARNING_RATE = 1e-3
