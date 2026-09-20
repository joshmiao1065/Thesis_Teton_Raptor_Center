# Source this before running BirdNET on an NVIDIA GPU (TensorFlow does not find the pip-installed
# CUDA libraries on its own):
#     source code/environments/birdnet_gpu.sh <path to the BirdNET venv>
# Then use the SavedModel backend, one worker, and a GPU device:
#     model = birdnet.load("acoustic", "2.4", "pb"); model.predict(files, n_workers=1, device="GPU:0", ...)
# (the default "tf" backend is CPU-only, and one worker per CPU thread exhausts GPU memory).
# The first GPU call may spend about a minute compiling kernels; the cache is reused afterwards.
VENV="${1:?path to the BirdNET venv}"
export LD_LIBRARY_PATH="$(ls -d "$VENV"/lib/python*/site-packages/nvidia/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}"
export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-4294967296}"
