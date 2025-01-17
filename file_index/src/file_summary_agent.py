"""
Agent for generating structured summaries of Python source files.
"""
import ast
from pathlib import Path
from typing import List, Dict, Any

class FileSummaryAgent:
    """Analyzes Python source files and generates structured summaries."""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.tree = None
        self._parse_file()
        
    def _parse_file(self):
        """Parse the Python file into an AST."""
        with open(self.file_path, 'r') as f:
            content = f.read()
        self.tree = ast.parse(content)
            
    def get_imports(self) -> List[str]:
        """Extract all import statements."""
        imports = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(name.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for name in node.names:
                    imports.append(f"{module}.{name.name}")
        return sorted(imports)
        
    def get_definitions(self) -> Dict[str, List[str]]:
        """Get all function and class definitions."""
        defs = {
            'functions': [],
            'classes': [],
            'methods': []
        }
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                defs['classes'].append(node.name)
                # Check for methods
                for subnode in node.body:
                    if isinstance(subnode, ast.FunctionDef):
                        if subnode.name != '__init__':
                            defs['methods'].append(f"{node.name}.{subnode.name}")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name != '__init__':
                    defs['functions'].append(node.name)

        # Remove any functions that are also methods
        defs['functions'] = [f for f in defs['functions'] if not \
                any([f in x for x in defs['methods']])
                             ]
                
        return {k: sorted(v) for k, v in defs.items()}
        
    def summarize_functionality(self) -> List[str]:
        """Generate bullet points summarizing the file's functionality."""
        # Extract docstrings
        bullets = []
        
        # Module docstring
        if (ast.get_docstring(self.tree)):
            bullets.append(ast.get_docstring(self.tree).split('\n')[0])
            
        # Class and function docstrings
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                doc = ast.get_docstring(node)
                if doc:
                    # bullets.append(f"- {node.name}: {doc.split('\n')[0]}")
                    bullets.append(f"- {node.name}: {doc}")
                    
        return bullets
        
    def generate_summary(self) -> Dict[str, Any]:
        """Generate complete file summary."""
        return {
            'imports': self.get_imports(),
            'definitions': self.get_definitions(),
            'functionality': self.summarize_functionality()
        }
