import json
import argparse
from pathlib import Path

def detect_honeypots(input_file, output_file):
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    print(f"Loading raw candidates from {input_path} to detect honeypots...")
    honeypot_ids = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        
        if first_char == '[':
            candidates = json.load(f)
        else:
            candidates = [json.loads(line) for line in f if line.strip()]
            
        for record in candidates:
            cid = record['candidate_id']
            is_honeypot = False
            
            # Simple heuristic 1: 8 years experience at a company founded 3 years ago
            # We don't have company found date, but we can check if duration_months > age of candidate (if age existed)
            # Or if total duration_months across career > years_of_experience * 12 significantly
            yoe_months = record.get('profile', {}).get('years_of_experience', 0) * 12
            career_history = record.get('career_history', [])
            total_career_months = sum([r.get('duration_months', 0) for r in career_history])
            
            if total_career_months > yoe_months + 60: # 5 years discrepancy
                is_honeypot = True
                
            # Simple heuristic 2: "expert" proficiency in skills with 0 years used
            skills = record.get('skills', [])
            for skill in skills:
                if skill.get('proficiency') == 'expert' and skill.get('duration_months', 0) == 0:
                    is_honeypot = True
                    break
                    
            if is_honeypot:
                honeypot_ids.append(cid)
                
    print(f"Detected {len(honeypot_ids)} honeypot profiles.")
    print(f"Saving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(honeypot_ids, f, indent=2)
        
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../../data/raw/sample_candidates.json", help="Path to raw candidates JSON/JSONL")
    parser.add_argument("--output", default="../../data/artifacts/honeypot_flags.json", help="Output path for JSON file")
    args = parser.parse_args()
    
    detect_honeypots(args.input, args.output)
