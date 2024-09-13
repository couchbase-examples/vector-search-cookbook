import os
from nbconvert import MarkdownExporter
from nbformat import read
from glob import glob

# todo: move these
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    OKGREENH = '\x1b[1;33;42m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    FAILH = '\x1b[1;33;41m'
    ENDC = '\033[0m'
    ENDH = '\x1b[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Paths
project_dir = '../../'
markdown_dir = 'output_md/'

# Ensure output directory exists
os.makedirs(markdown_dir, exist_ok=True)

# Initialize nbconvert's Markdown exporter
exporter = MarkdownExporter()

# Iterate over all notebooks in the directory
for notebook_path in glob(f'{project_dir}/**/*.ipynb', recursive=True):
    # Check for a frontmatter doc
    frontmatter_path = f'{os.path.dirname(notebook_path)}/frontmatter.md'

    if os.path.isfile(frontmatter_path):
        # Read the frontmatter content
        with open(frontmatter_path, 'r', encoding='utf-8') as frontmatter_file:
            frontmatter_content = frontmatter_file.read()
    else:
        # Skip the notebook if no frontmatter.md is found
        print(f'{bcolors.FAILH} FAIL {bcolors.ENDH} - {bcolors.FAIL}Skipping conversion for {notebook_path} \n\t Reason: frontmatter.md not found.{bcolors.ENDC}')
        continue

    # Read the notebook content
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = read(f, as_version=4)

    # Convert the notebook to markdown
    body, resources = exporter.from_notebook_node(notebook)

    # Combine frontmatter with notebook markdown
    combined_markdown = f"{frontmatter_content}\n\n{body}"

    # Generate the output file path
    notebook_name = os.path.splitext(os.path.basename(notebook_path))[0]
    markdown_file = os.path.join(markdown_dir, f'{notebook_name}.md')

    # Write the combined markdown to a file
    with open(markdown_file, 'w', encoding='utf-8') as f:
        f.write(combined_markdown)

    print(f'{bcolors.OKGREENH}  OK  {bcolors.ENDH} - Converted \n\t\t{bcolors.HEADER}{notebook_path}{bcolors.ENDC} to \n\t\t{bcolors.HEADER}{markdown_file}{bcolors.ENDC} \n\t with frontmatter.')

