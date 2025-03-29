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

GESTURES_SIZE = 15

def label_to_number(label):
    if label == "G1":
        return 0
    elif label == "G2":
        return 1
    elif label == "G3":
        return 2
    elif label == "G4":
        return 3
    elif label == "G5":
        return 4
    elif label == "G6":
        return 5
    elif label == "G7":
        return 6
    elif label == "G8":
        return 7
    elif label == "G9":
        return 8
    elif label == "G10":
        return 9
    elif label == "G11":
        return 10
    elif label == "G12":
        return 11
    elif label == "G13":
        return 12
    elif label == "G14":
        return 13
    elif label == "G15":
        return 14
    
def number_to_label(number):
    if number == 0:
        return "G1"
    elif number == 1:
        return "G2"
    elif number == 2:
        return "G3"
    elif number == 3:
        return "G4"
    elif number == 4:
        return "G5"
    elif number == 5:
        return "G6"
    elif number == 6:
        return "G7"
    elif number == 7:
        return "G8"
    elif number == 8:
        return "G9"
    elif number == 9:
        return "G10"
    elif number == 10:
        return "G11"
    elif number == 11:
        return "G12"
    elif number == 12:
        return "G13"
    elif number == 13:
        return "G14"
    elif number == 14:
        return "G15"

# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"