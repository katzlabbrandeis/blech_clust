"""
Code to create a file with all unique imports from function_analysis.json

This script reads the function_analysis.json file and generates a Python file
that contains all unique import statements found in the codebase.
"""

import os
import json
import sys

def main():
    """
    Main function to generate a file with all unique imports.
    """
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the function analysis JSON file
    json_file = os.path.join(script_dir, "function_analysis.json")
    
    # Path for the output file
    output_file = os.path.join(script_dir, "all_unique_imports.py")
    
    # Check if the JSON file exists
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found", file=sys.stderr)
        return 1
    
    # Load the JSON data
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}", file=sys.stderr)
        return 1
    
    # Extract all imports
    imports = data.get('imports', [])
    
    if not imports:
        print("No imports found in the JSON file", file=sys.stderr)
        return 1
    
    # Sort imports for better organization
    imports.sort()
    
    # Generate the Python file with import statements
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('"""\nAutomatically generated file containing all unique imports found in the codebase.\n')
        f.write('This file is for testing dependency resolution and import availability.\n"""\n\n')
        f.write('import os\n')
        f.write('import sys\n\n')
        f.write('# Add parent directory (blech_clust) to path\n')
        f.write('sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))\n\n')
        f.write('# Standard library imports\n')
        
        # Process each import
        for imp in imports:
            # Skip empty imports
            if not imp.strip():
                continue
                
            parts = imp.split('.')
            
            # Handle different import formats
            if '*' in imp:
                # Handle wildcard imports like "module.*"
                module = imp.replace('.*', '')
                f.write(f'try:\n    from {module} import *\nexcept ImportError:\n    print(f"Failed to import: {imp}")\n\n')
            elif len(parts) == 1:
                # Simple import like "numpy"
                f.write(f'try:\n    import {imp}\nexcept ImportError:\n    print(f"Failed to import: {imp}")\n\n')
            else:
                # From import like "numpy.array"
                module = '.'.join(parts[:-1])
                item = parts[-1]
                f.write(f'try:\n    from {module} import {item}\nexcept ImportError:\n    print(f"Failed to import: {imp}")\n\n')
    
    print(f"Generated {output_file} with {len(imports)} unique imports")
    return 0

if __name__ == "__main__":
    sys.exit(main())
