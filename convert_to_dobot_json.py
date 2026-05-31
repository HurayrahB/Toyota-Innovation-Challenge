import json
import os

def convert_workspace_to_json(output_filename="dobot_workspace.json"):
    py_files = []
    
    # Walk through the directory to find all .py files
    for root, dirs, files in os.walk("."):
        # Skip hidden directories like .git and cache directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            if file.endswith(".py") and file != os.path.basename(__file__):
                # We store the relative path to read it, but we'll use the basename for the tab
                py_files.append(os.path.join(root, file))
                
    # Sort files to have a consistent order
    py_files.sort()

    code_list = []
    tabs_list = []

    for py_file in py_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            code_list.append(content)
            
            # Extract the filename to use as the tab name in Dobot Lab
            filename = os.path.basename(py_file)
            tabs_list.append(filename)
            print(f"Added {py_file} as tab '{filename}'")
        except Exception as e:
            print(f"Failed to read {py_file}: {e}")

    # Construct the JSON structure required by Dobot Lab
    dobot_json = {
        "script": {
            "code": code_list,
            "tabs": tabs_list
        }
    }

    # Write the JSON to the output file
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(dobot_json, f, indent=2)
        
    print(f"\nSuccessfully converted {len(py_files)} files into {output_filename}")

if __name__ == "__main__":
    convert_workspace_to_json()
