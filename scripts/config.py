from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "datasets" 

# Ensure the dataset directory exists
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found in {DATA_PATH}. "
        "Make sure you downloaded the dataset and placed it in the correct folder."
    )

KINEMATICS_FEATURES_SIZE = 76

GESTURES_SIZE = 12

def label_to_number(label):
    if label == "G1":
        return 1
    elif label == "G2":
        return 2
    elif label == "G3":
        return 3
    elif label == "G4":
        return 4
    elif label == "G5":
        return 5
    elif label == "G6":
        return 6
    elif label == "G7":
        return 7
    elif label == "G8":
        return 8
    elif label == "G9":
        return 9
    elif label == "G10":
        return 10
    elif label == "G11":
        return 11
    elif label == "G12":
        return 12
    elif label == "G13":
        return 13
    elif label == "G14":
        return 14
    elif label == "G15":
        return 15