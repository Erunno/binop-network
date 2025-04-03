import os
import sys
import time
import subprocess
import datetime
from pathlib import Path
import concurrent.futures

# Configuration constants
LOG_DIR = Path("logs")
PROCESS_COUNT = 4  # Number of parallel processes
MAX_STEPS = 1_000_000   # Maximum steps per process

# Parameter variations
LAYER_CONFIGS = [
    [20, 15, 9, 5, 3, 2, 1],
    [15, 10, 7, 5, 3, 1],
    [25, 18, 12, 8, 5, 3, 1],
    [12, 9, 7, 5, 3, 2, 1],
    [30, 20, 15, 10, 5, 1],
]

MIX_UP_COEFFICIENTS = [0.05, 0.1, 0.15, 0.2]

STOCHASTIC_CONFIGS = [
    [(2, 10000), (3, 8000), (4, 5000)],
    [(2, 5000), (3, 4000), (4, 3000)],
    [(2, 15000), (3, 10000)],
    [(3, 12000), (4, 8000), (5, 4000)],
]

SEED_BASE = 1000  # Base seed value (process_id will be added to this)

def create_log_directory():
    """Create a uniquely named log directory based on current time"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = LOG_DIR / f"experiment-{timestamp}"
    log_path.mkdir(parents=True, exist_ok=True)
    return log_path

def build_command(process_id, log_path):
    """Build the command for a specific process with unique parameters"""
    # Select parameters for this process (cycling through available options)
    layers = LAYER_CONFIGS[process_id % len(LAYER_CONFIGS)]
    mix_up = MIX_UP_COEFFICIENTS[process_id % len(MIX_UP_COEFFICIENTS)]
    stochastic_config = STOCHASTIC_CONFIGS[process_id % len(STOCHASTIC_CONFIGS)]
    seed = SEED_BASE + process_id
    
    # Build layers argument
    layers_arg = " ".join(str(layer) for layer in layers)
    
    # Build distance samples argument
    distance_samples_arg = " ".join(f"{d} {s}" for d, s in stochastic_config)
    
    # Fix the path to main.py - go up one directory level from cluster-experiments
    main_py_path = Path(__file__).parent.parent / "python-implementation" / "main.py"
    
    # Construct full command
    cmd = [
        sys.executable,  # Use the current Python interpreter
        str(main_py_path),
        "--layers", *map(str, layers),
        "--mix-up", str(mix_up),
        "--seed", str(seed),
        "--max-steps", str(MAX_STEPS),
        "--distance-samples", *distance_samples_arg.split()
    ]
    
    return cmd

def run_process(process_id, log_path):
    """Run a single network exploration process"""
    cmd = build_command(process_id, log_path)
    
    # Create log file
    log_file = log_path / f"process_{process_id}.log"
    
    print(f"Starting process {process_id} with parameters:")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log file: {log_file}")
    
    # Run the process and redirect output to log file
    with open(log_file, 'w') as log:
        process = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True
        )
        return process

def main():
    # Create log directory
    log_path = create_log_directory()
    print(f"Created log directory: {log_path}")
    
    # Save experiment configuration
    with open(log_path / "experiment_config.txt", 'w') as f:
        f.write(f"Date: {datetime.datetime.now().isoformat()}\n")
        f.write(f"Process count: {PROCESS_COUNT}\n")
        f.write(f"Max steps: {MAX_STEPS}\n\n")
        
        f.write("Layer configurations:\n")
        for i, config in enumerate(LAYER_CONFIGS):
            f.write(f"  {i}: {config}\n")
        
        f.write("\nMix-up coefficients:\n")
        for i, coef in enumerate(MIX_UP_COEFFICIENTS):
            f.write(f"  {i}: {coef}\n")
        
        f.write("\nStochastic configs:\n")
        for i, config in enumerate(STOCHASTIC_CONFIGS):
            f.write(f"  {i}: {config}\n")
    
    # Start processes
    processes = []
    for i in range(PROCESS_COUNT):
        process = run_process(i, log_path)
        processes.append(process)
    
    print(f"Started {PROCESS_COUNT} processes")
    
    # Wait for all processes to complete
    for i, process in enumerate(processes):
        return_code = process.wait()
        print(f"Process {i} completed with return code {return_code}")
    
    print("All processes completed")

if __name__ == "__main__":
    main()
