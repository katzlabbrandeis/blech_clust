#!/usr/bin/env python3
"""
Script to capture all unique function calls in the blech_clust library.
This script walks through the repository, analyzes Python files,
and extracts function definitions and calls.
"""

import os
import ast
import sys
from collections import defaultdict
import json
from typing import Dict, List, Set, Tuple


class FunctionVisitor(ast.NodeVisitor):
    """AST visitor to extract function definitions and calls."""
    
    def __init__(self):
        self.defined_functions = set()
        self.called_functions = set()
        self.method_calls = set()
        self.imports = set()
        self.current_class = None
        self.current_function = None
        
    def visit_FunctionDef(self, node):
        """Record function definitions."""
        if self.current_class:
            self.defined_functions.add(f"{self.current_class}.{node.name}")
        else:
            self.defined_functions.add(node.name)
        
        prev_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = prev_function
        
    def visit_ClassDef(self, node):
        """Record class definitions and their methods."""
        prev_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = prev_class
        
    def visit_Call(self, node):
        """Record function calls."""
        if isinstance(node.func, ast.Name):
            # Direct function call like func()
            self.called_functions.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            # Method call like obj.method()
            if isinstance(node.func.value, ast.Name):
                self.method_calls.add(f"{node.func.value.id}.{node.func.attr}")
        
        self.generic_visit(node)
        
    def visit_Import(self, node):
        """Record imports."""
        for name in node.names:
            self.imports.add(name.name)
            
    def visit_ImportFrom(self, node):
        """Record from imports."""
        if node.module:
            for name in node.names:
                self.imports.add(f"{node.module}.{name.name}")


def analyze_file(file_path: str) -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
    """
    Analyze a Python file to extract function definitions and calls.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        Tuple of (defined_functions, called_functions, method_calls, imports)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        visitor = FunctionVisitor()
        visitor.visit(tree)
        
        return (
            visitor.defined_functions,
            visitor.called_functions,
            visitor.method_calls,
            visitor.imports
        )
    except (SyntaxError, UnicodeDecodeError, IsADirectoryError, PermissionError) as e:
        print(f"Error analyzing {file_path}: {e}", file=sys.stderr)
        return set(), set(), set(), set()


def find_python_files(start_dir: str) -> List[str]:
    """
    Find all Python files in the repository.
    
    Args:
        start_dir: Starting directory
        
    Returns:
        List of Python file paths
    """
    python_files = []
    
    for root, _, files in os.walk(start_dir):
        # Skip virtual environments and hidden directories
        if ('venv' in root.split(os.sep) or 
            '.env' in root.split(os.sep) or 
            '__pycache__' in root.split(os.sep) or
            any(part.startswith('.') for part in root.split(os.sep))):
            continue
            
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
                
    return python_files


def analyze_repository(repo_dir: str) -> Dict:
    """
    Analyze the entire repository to find all function definitions and calls.
    
    Args:
        repo_dir: Repository root directory
        
    Returns:
        Dictionary with analysis results
    """
    all_defined_functions = set()
    all_called_functions = set()
    all_method_calls = set()
    all_imports = set()
    
    file_functions = {}
    
    python_files = find_python_files(repo_dir)
    print(f"Found {len(python_files)} Python files to analyze")
    
    for file_path in python_files:
        rel_path = os.path.relpath(file_path, repo_dir)
        defined, called, methods, imports = analyze_file(file_path)
        
        file_functions[rel_path] = {
            "defined": list(defined),
            "called": list(called),
            "methods": list(methods),
            "imports": list(imports)
        }
        
        all_defined_functions.update(defined)
        all_called_functions.update(called)
        all_method_calls.update(methods)
        all_imports.update(imports)
    
    # Find potentially undefined functions (called but not defined in the codebase)
    undefined_functions = all_called_functions - all_defined_functions
    
    # Remove standard library and imported functions from undefined
    standard_lib_funcs = {
        'print', 'len', 'range', 'open', 'str', 'int', 'float', 'list', 
        'dict', 'set', 'tuple', 'sum', 'min', 'max', 'sorted', 'enumerate',
        'zip', 'map', 'filter', 'any', 'all', 'abs', 'round', 'isinstance',
        'issubclass', 'hasattr', 'getattr', 'setattr', 'delattr', 'dir',
        'vars', 'type', 'id', 'hash', 'help', 'input', 'super'
    }
    
    undefined_functions = {f for f in undefined_functions 
                          if f not in standard_lib_funcs and 
                          not any(f.startswith(imp.split('.')[0]) for imp in all_imports)}
    
    return {
        "defined_functions": sorted(list(all_defined_functions)),
        "called_functions": sorted(list(all_called_functions)),
        "method_calls": sorted(list(all_method_calls)),
        "imports": sorted(list(all_imports)),
        "undefined_functions": sorted(list(undefined_functions)),
        "file_functions": file_functions
    }


def main():
    """Main function to run the analysis."""
    # Get repository root (assuming this script is in the repo)
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print(f"Analyzing repository at: {repo_dir}")
    results = analyze_repository(repo_dir)
    
    # Save results to JSON file
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              "function_analysis.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\nAnalysis Summary:")
    print(f"- Defined functions: {len(results['defined_functions'])}")
    print(f"- Called functions: {len(results['called_functions'])}")
    print(f"- Method calls: {len(results['method_calls'])}")
    print(f"- Imports: {len(results['imports'])}")
    print(f"- Potentially undefined functions: {len(results['undefined_functions'])}")
    print(f"\nDetailed results saved to: {output_file}")
    
    # Create a simple HTML report for better visualization
    create_html_report(results, os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                           "function_analysis.html"))


def create_html_report(results: Dict, output_file: str):
    """
    Create an HTML report from the analysis results.
    
    Args:
        results: Analysis results dictionary
        output_file: Path to save the HTML report
    """
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Blech Clust Function Analysis</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
        h1, h2, h3 { color: #333; }
        .section { margin-bottom: 30px; }
        .file-section { margin-bottom: 20px; border-bottom: 1px solid #eee; padding-bottom: 10px; }
        .function-list { max-height: 300px; overflow-y: auto; background: #f9f9f9; padding: 10px; border-radius: 5px; }
        .collapsible { cursor: pointer; padding: 10px; background: #f1f1f1; border: none; text-align: left; width: 100%; }
        .active, .collapsible:hover { background-color: #ddd; }
        .content { display: none; padding: 10px; overflow: hidden; background-color: #f9f9f9; }
    </style>
</head>
<body>
    <h1>Blech Clust Function Analysis</h1>
    
    <div class="section">
        <h2>Summary</h2>
        <p>Defined functions: """ + str(len(results['defined_functions'])) + """</p>
        <p>Called functions: """ + str(len(results['called_functions'])) + """</p>
        <p>Method calls: """ + str(len(results['method_calls'])) + """</p>
        <p>Imports: """ + str(len(results['imports'])) + """</p>
        <p>Potentially undefined functions: """ + str(len(results['undefined_functions'])) + """</p>
    </div>
    
    <div class="section">
        <h2>Defined Functions</h2>
        <div class="function-list">
            <ul>
"""
    
    for func in results['defined_functions']:
        html += f"                <li>{func}</li>\n"
    
    html += """            </ul>
        </div>
    </div>
    
    <div class="section">
        <h2>Potentially Undefined Functions</h2>
        <div class="function-list">
            <ul>
"""
    
    for func in results['undefined_functions']:
        html += f"                <li>{func}</li>\n"
    
    html += """            </ul>
        </div>
    </div>
    
    <div class="section">
        <h2>Files Analysis</h2>
"""
    
    for file_path, file_data in results['file_functions'].items():
        html += f"""        <button class="collapsible">{file_path}</button>
        <div class="content">
            <h3>Defined Functions</h3>
            <ul>
"""
        
        for func in file_data['defined']:
            html += f"                <li>{func}</li>\n"
        
        html += """            </ul>
            <h3>Called Functions</h3>
            <ul>
"""
        
        for func in file_data['called']:
            html += f"                <li>{func}</li>\n"
        
        html += """            </ul>
            <h3>Method Calls</h3>
            <ul>
"""
        
        for method in file_data['methods']:
            html += f"                <li>{method}</li>\n"
        
        html += """            </ul>
        </div>
"""
    
    html += """    </div>

    <script>
        var coll = document.getElementsByClassName("collapsible");
        for (var i = 0; i < coll.length; i++) {
            coll[i].addEventListener("click", function() {
                this.classList.toggle("active");
                var content = this.nextElementSibling;
                if (content.style.display === "block") {
                    content.style.display = "none";
                } else {
                    content.style.display = "block";
                }
            });
        }
    </script>
</body>
</html>
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"HTML report created at: {output_file}")


if __name__ == "__main__":
    main()
