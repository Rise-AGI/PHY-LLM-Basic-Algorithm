#!/usr/bin/env python
import sys

print("Step 1: Checking Python...")
print(f"Python version: {sys.version}")

print("\nStep 2: Checking torch...")
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU count: {torch.cuda.device_count()}")
except Exception as e:
    print(f"Error importing torch: {e}")
    sys.exit(1)

print("\nStep 3: Checking transformers...")
try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
except Exception as e:
    print(f"Error importing transformers: {e}")
    sys.exit(1)

print("\nStep 4: Checking peft...")
try:
    import peft
    print(f"PEFT version: {peft.__version__}")
except Exception as e:
    print(f"Error importing peft: {e}")
    print("Please run: pip install peft")
    sys.exit(1)

print("\nStep 5: Checking datasets...")
try:
    import datasets
    print(f"Datasets version: {datasets.__version__}")
except Exception as e:
    print(f"Error importing datasets: {e}")
    print("Please run: pip install datasets")
    sys.exit(1)

print("\nStep 6: Checking accelerate...")
try:
    import accelerate
    print(f"Accelerate version: {accelerate.__version__}")
except Exception as e:
    print(f"Error importing accelerate: {e}")
    print("Please run: pip install accelerate")
    sys.exit(1)

print("\nStep 7: Testing GPU operation...")
try:
    x = torch.randn(10, 10, device='cuda')
    y = torch.matmul(x, x)
    print("GPU tensor operation: SUCCESS")
    del x, y
    torch.cuda.empty_cache()
except Exception as e:
    print(f"GPU operation failed: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("All checks passed! Environment is ready.")
print("="*50)