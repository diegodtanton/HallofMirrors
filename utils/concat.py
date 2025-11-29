import os
import glob
from pathlib import Path
import sys

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

# Directories to completely ignore (not in tree, not in content)
IGNORE_DIRS = [
    '__pycache__',
    '.git',
    '.vscode',
    '.idea',
    'logs',
    'venv',
    'env',
    'node_modules',
    'analysis_results'
]

# Files to ignore (not in tree, not in content)
IGNORE_FILES = [
    '.DS_Store',
    'mirrors_project.zip'
]

# Files/Dirs to show in Tree, but EXCLUDE content from Markdown
EXCLUDE_CONTENT_PATTERNS = [
    'concat.py',
    'clean.py',
    'concat.md',
    'mirrors_summary', # CSVs
    'gauge_analysis',  # CSVs
    '.png',
    '.jpg',
    '.pt'              # Checkpoints
]

# Readable extensions for content inclusion
READABLE_EXTENSIONS = [
    '.py', '.html', '.js', '.md', '.yaml', 
    '.yml', '.jinja', '.css', '.txt', '.sh', 
    '.json',
]

# ---------------------------------------------------------
# SCRIPT
# ---------------------------------------------------------

def get_project_root():
    # Assumes script is in /utils/concat.py
    # Root is one level up
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)

OUTPUT_FILE = os.path.join(get_project_root(), 'utils', 'concat.md')

def read_file_robust(file_path):
    """
    Tries multiple encodings to handle weird Windows/PowerShell files.
    """
    encodings = ['utf-8', 'utf-16', 'cp1252', 'latin-1']
    
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                content = f.read()
                # If utf-16 decoding works but yields garbage, it might look like Chinese characters
                # but usually Python raises error. This is generally safe.
                return content
        except UnicodeError:
            continue
    
    return f"Error: Could not decode file with encodings: {encodings}"

def should_process_file(file_path, project_root):
    """
    Decides if a file belongs in the 'Project Structure' tree.
    """
    rel_path = os.path.relpath(file_path, project_root)
    parts = rel_path.split(os.sep)

    # 1. Check ignored directories
    for part in parts:
        if part in IGNORE_DIRS:
            return False
    
    # 2. Check ignored files
    if os.path.basename(file_path) in IGNORE_FILES:
        return False

    return True

def get_all_files():
    """
    Walks the directory to get ALL relevant files for the tree structure.
    """
    project_root = get_project_root()
    all_files = []

    for root, dirs, files in os.walk(project_root):
        # Modify dirs in-place to skip walking ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            full_path = os.path.join(root, file)
            if should_process_file(full_path, project_root):
                rel_path = os.path.relpath(full_path, project_root)
                all_files.append(rel_path)

    # Sort for consistent output
    all_files.sort()
    return all_files

def print_file_tree(files):
    """Prints the ASCII tree."""
    tree = {}
    for path in files:
        parts = path.split(os.sep)
        current = tree
        for part in parts:
            current = current.setdefault(part, {})

    def _print_node(node, prefix=''):
        keys = sorted(node.keys())
        for i, key in enumerate(keys):
            is_last = (i == len(keys) - 1)
            connector = '└── ' if is_last else '├── '
            print(f"{prefix}{connector}{key}")
            
            extension = prefix + ('    ' if is_last else '│   ')
            if node[key]:  # If it has children, it's a directory
                _print_node(node[key], extension)

    print("Project Structure:")
    print("=================")
    _print_node(tree)
    print("=================\n")

def create_concatenated_file(files):
    project_root = get_project_root()

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
        # Header & Tree
        out.write("# Concatenated Project Files\n\n")
        out.write("```\n")
        
        # Capture print output to write to file
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            print_file_tree(files)
        out.write(f.getvalue())
        
        out.write("```\n\n")

        # File Contents
        for rel_path in files:
            full_path = os.path.join(project_root, rel_path)
            filename = os.path.basename(full_path)
            
            # Check Exclusion Patterns (Content Only)
            skip_content = False
            
            # 1. Explicit filename skips
            if filename in EXCLUDE_CONTENT_PATTERNS:
                skip_content = True
            
            # 2. Extension skips (binary files)
            ext = os.path.splitext(filename)[1].lower()
            if ext not in READABLE_EXTENSIONS:
                skip_content = True

            # 3. Output file self-exclusion
            if os.path.abspath(full_path) == os.path.abspath(OUTPUT_FILE):
                skip_content = True

            if skip_content:
                continue

            # Write Content
            out.write(f"## File: {rel_path}\n\n")
            
            code_block_type = ext.lstrip('.')
            if code_block_type == "": code_block_type = "txt"
            
            out.write(f"```{code_block_type}\n")
            out.write(read_file_robust(full_path))
            out.write("\n```\n\n")

if __name__ == "__main__":
    print(f"Scanning root: {get_project_root()}")
    files = get_all_files()
    print_file_tree(files)
    create_concatenated_file(files)
    print(f"Done! Output saved to: {OUTPUT_FILE}")