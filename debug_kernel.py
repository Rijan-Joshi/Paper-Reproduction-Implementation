import sys
print(f"Python executable: {sys.executable}")
try:
    import ipykernel
    print(f"ipykernel imported successfully. Version: {ipykernel.__version__}")
    import zmq
    print(f"zmq imported successfully. Version: {zmq.__version__}")
except Exception as e:
    print(f"Error importing: {e}")
