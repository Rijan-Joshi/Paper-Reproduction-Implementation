import sys
print(f"Python version: {sys.version}")
try:
    import torch
    print(f"PyTorch imported successfully. Version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"Failed to import PyTorch: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
