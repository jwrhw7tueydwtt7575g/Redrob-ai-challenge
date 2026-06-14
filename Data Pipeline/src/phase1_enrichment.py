import json
import argparse
import sys
from pathlib import Path

def build_career_narrative(profile, career_history):
    lines = []
    
    # Profile summary
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")
    yoe = profile.get("years_of_experience", 0)
    lines.append(f"Candidate is a {headline} with {yoe} years of experience.")
    if summary:
        lines.append(f"Professional Summary: {summary}")
        
    lines.append("Career History:")
    for role in career_history:
        company = role.get("company", "Unknown Company")
        title = role.get("title", "Unknown Title")
        duration = role.get("duration_months", 0)
        desc = role.get("description", "")
        
        lines.append(f"- Worked at {company} as {title} for {duration} months. {desc}")
        
    return "\n".join(lines)

def build_skill_narrative(skills):
    if not skills:
        return "No skills listed."
        
    lines = ["Skills:"]
    for skill in skills:
        name = skill.get("name", "Unknown Skill")
        proficiency = skill.get("proficiency", "unknown proficiency").capitalize()
        duration = skill.get("duration_months", 0)
        endorsements = skill.get("endorsements", 0)
        
        lines.append(f"- {proficiency} {name} - {duration} months used, {endorsements} endorsements")
        
    return "\n".join(lines)

def build_behavioral_text(signals):
    if not signals:
        return "No behavioral signals available."
        
    lines = ["Behavioral Signals:"]
    
    last_active = signals.get("last_active_date", "Unknown")
    lines.append(f"- Last active on {last_active}.")
    
    rr = signals.get("recruiter_response_rate", 0)
    lines.append(f"- Responds to {int(rr * 100)}% of recruiter messages.")
    
    github = signals.get("github_activity_score", -1)
    if github >= 0:
        lines.append(f"- GitHub activity score is {github}/100.")
    else:
        lines.append("- No GitHub account linked.")
        
    np = signals.get("notice_period_days", 0)
    lines.append(f"- Notice period is {np} days.")
    
    views = signals.get("profile_views_received_30d", 0)
    apps = signals.get("applications_submitted_30d", 0)
    lines.append(f"- Received {views} profile views and submitted {apps} applications in the last 30 days.")
    
    salary = signals.get("expected_salary_range_inr_lpa", {})
    if salary:
        lines.append(f"- Expected salary range is {salary.get('min', 0)} to {salary.get('max', 0)} LPA INR.")
        
    mode = signals.get("preferred_work_mode", "any")
    relocate = "willing" if signals.get("willing_to_relocate") else "unwilling"
    lines.append(f"- Prefers {mode} work and is {relocate} to relocate.")
    
    return "\n".join(lines)

def process_candidate(candidate):
    cid = candidate.get("candidate_id", "UNKNOWN_ID")
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    
    career_text = build_career_narrative(profile, career)
    skill_text = build_skill_narrative(skills)
    behavioral_text = build_behavioral_text(signals)
    
    enriched_text = f"--- CAREER NARRATIVE ---\n{career_text}\n\n--- SKILL NARRATIVE ---\n{skill_text}\n\n--- BEHAVIORAL SIGNALS ---\n{behavioral_text}"
    
    return {
        "candidate_id": cid,
        "enriched_text": enriched_text
    }

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Text Enrichment")
    parser.add_argument("--input", required=True, help="Input file (JSON or JSONL)")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found.")
        sys.exit(1)
        
    print(f"Processing {input_path}...")
    
    count = 0
    with open(output_path, 'w', encoding='utf-8') as outfile:
        # Check if JSON or JSONL
        if input_path.suffix.lower() == '.jsonl':
            with open(input_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    if not line.strip(): continue
                    candidate = json.loads(line)
                    enriched = process_candidate(candidate)
                    outfile.write(json.dumps(enriched) + '\n')
                    count += 1
        else:
            # Assume it's a JSON array
            with open(input_path, 'r', encoding='utf-8') as infile:
                candidates = json.load(infile)
                for candidate in candidates:
                    enriched = process_candidate(candidate)
                    outfile.write(json.dumps(enriched) + '\n')
                    count += 1
                    
    print(f"Finished processing {count} records. Output written to {output_path}")

if __name__ == "__main__":
    main()
