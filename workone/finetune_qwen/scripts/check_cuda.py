import torch
import subprocess

def check_cuda():
    print("=" * 60)
    print("CUDA and GPU Environment Check")
    print("=" * 60)
    
    # PyTorch CUDA info
    print(f"\n[PyTorch Info]")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version (PyTorch): {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        print(f"\n[GPU Details]")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Total memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Multi-processor count: {props.multi_processor_count}")
            print(f"  Compute capability: {props.major}.{props.minor}")
    
    # nvcc check
    print(f"\n[nvcc Check]")
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("nvcc not found in PATH")
    except FileNotFoundError:
        print("nvcc command not found")
    
    # Environment variables
    print(f"\n[Environment Variables]")
    import os
    for var in ['CUDA_HOME', 'CUDA_PATH', 'CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH']:
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {value}")
    
    # Test tensor operation
    if torch.cuda.is_available():
        print(f"\n[Tensor Operation Test]")
        try:
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.randn(1000, 1000, device='cuda')
            z = torch.matmul(x, y)
            print("GPU tensor operation: SUCCESS")
            del x, y, z
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"GPU tensor operation: FAILED - {e}")
    
    print("=" * 60)


if __name__ == "__main__":
    check_cuda()