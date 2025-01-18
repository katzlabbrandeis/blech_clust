"""
Iterate through files in a directory and generate a summary of each file.
"""
import os
import sys
import json
from pathlib import Path
from glob import glob
from file_summary_agent import FileSummaryAgent

def main():
    file_path = Path(__file__).resolve() 
    blech_dir = file_path.parents[2]
    output_dir = file_path.parents[1] / 'data' / 'summaries'
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Base dir files
    iter_dirs = [
            blech_dir,
            blech_dir / 'emg',
            blech_dir / 'emg' / 'utils',
            blech_dir / 'utils',
            blech_dir / 'utils' / 'ephys_data',
            blech_dir / 'utils' / 'qa_utils',
            ]

    # Iterate through directories and find all python files
    file_list = []
    for dir in iter_dirs:
        for file in dir.glob('*.py'):
            file_list.append(file)
    sorted_files = sorted(file_list)
    
    # Generate summaries for each file
    summary_list = []
    for file_path in sorted_files:
        try:
            # Generate relative path for output file
            rel_path = file_path.relative_to(blech_dir)
            output_file = output_dir / f"{rel_path.stem}_summary.json"
            # output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate summary
            agent = FileSummaryAgent(file_path)
            summary = agent.generate_summary()
            summary_list.append(summary)
            
            # Save summary
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2)
                
            print(f"Generated summary for {rel_path}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}", file=sys.stderr)

    # Write out merged summary
    merged_out_path = output_dir.parents[0] / 'merged_summary.json'
    with open(merged_out_path, 'w') as f:
        json.dump(summary_list, f, indent=4)
    print(f"Merged summary written to file : {merged_out_path}")

if __name__ == "__main__":
    main()
