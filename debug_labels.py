
import traceback
from src.utils.config import TRAIN_LABELS
from src.data.labels import load_labels

print("TEST 1: Starting label-loader test", flush=True)
print(f"TEST 2: CSV path = {TRAIN_LABELS}", flush=True)

try:
    print("TEST 3: Calling load_labels()...", flush=True)

    labels, stats = load_labels(TRAIN_LABELS)

    print("TEST 4: Label loading completed!", flush=True)
    print("Stats:", stats, flush=True)
    print("Number of patients:", len(labels), flush=True)

except BaseException:
    traceback.print_exc()
    raise

print("TEST 5: Script completed successfully", flush=True)
