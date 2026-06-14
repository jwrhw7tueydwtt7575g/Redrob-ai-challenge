import subprocess
from pathlib import Path

def run_script(script_name, cwd):
    print(f"\n{'='*50}\nRunning {script_name}...\n{'='*50}")
    result = subprocess.run(["python3", script_name], cwd=cwd)
    if result.returncode != 0:
        print(f"Error: {script_name} failed with exit code {result.returncode}")
    else:
        print(f"Success: {script_name} completed.")

def main():
    src_dir = Path(__file__).parent / "phase2_precompute"
    
    scripts = [
        "track_a_bm25.py",
        "track_b_embeddings.py",
        "track_c_features.py",
        "interaction_builder.py",
        "silver_label_generator.py",
        "honeypot_detector.py"
    ]
    
    for script in scripts:
        run_script(script, cwd=src_dir)
        
    print("\nPhase 2 Precomputation Complete! All artifacts have been saved to data/artifacts/")

if __name__ == "__main__":
    main()
