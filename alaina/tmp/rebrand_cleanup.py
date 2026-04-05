import os
import re

def cleanup_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Remove legacy file path comments (e.g. # d:/quantly/quanta/...)
    # Matches lines starting with # followed by a path-like string
    cleaned = re.sub(r'^# [a-zA-Z]:/[^\n]+\n', '', content, flags=re.MULTILINE)

    # 2. Update Quanta references to ChatFlow (where appropriate)
    cleaned = cleaned.replace('Quanta Chat', 'ChatFlow Analysis')
    cleaned = cleaned.replace('Quanta AI', 'ChatFlow AI')
    
    # 3. Handle the stale quantalytics imports in base_tool.py specifically
    if 'base_tool.py' in filepath:
        # Remove the try-except block for quantalytics.performance
        cleaned = re.sub(r'# Performance helpers.*?_GLOBAL_CACHE = AdvancedCache\(\) if AdvancedCache is not None else None', 
                         '# Global cache disabled (Legacy performance module removed)\n_GLOBAL_CACHE = None', 
                         cleaned, flags=re.DOTALL)

    if cleaned != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        print(f"Cleaned: {filepath}")

def process_dir(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py') or file.endswith('.html'):
                cleanup_file(os.path.join(root, file))

if __name__ == "__main__":
    process_dir('f:/saiha/alaina/saiha')
