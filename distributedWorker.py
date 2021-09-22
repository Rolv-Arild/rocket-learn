from serious import make_worker, Necto, collate  # noqa
import sys
import torch

if not torch.cuda.is_available():
    sys.exit("Unable to train on your hardware, perhaps due to out-dated drivers or hardware age.")

name = "friendly-helper"

if len(sys.argv) >= 2:
    name = sys.argv[1]

print(name)

make_worker("89.162.46.245", name)
