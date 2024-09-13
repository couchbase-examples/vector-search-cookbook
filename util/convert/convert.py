import os
from nbconvert import MarkdownExporter
from nbformat import read
from glob import glob
from logger import Logger

# Paths
project_dir = "../../"
markdown_dir = "output_md/"

# Ensure output directory exists
os.makedirs(markdown_dir, exist_ok=True)

# Initialize nbconvert's Markdown exporter
exporter = MarkdownExporter()

# Iterate over all notebooks in the directory
for notebook_path in glob(f"{project_dir}/**/*.ipynb", recursive=True):
    # Check for a frontmatter doc
    frontmatter_path = f"{os.path.dirname(notebook_path)}/frontmatter.md"

    if os.path.isfile(frontmatter_path):
        # Read the frontmatter content
        with open(frontmatter_path, "r", encoding="utf-8") as frontmatter_file:
            frontmatter_content = frontmatter_file.read()
    else:
        # Skip the notebook if no frontmatter.md is found
        Logger.fail_conversion(notebook_path, "frontmatter.md not found")
        continue

    # Read the notebook content
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = read(f, as_version=4)

    # Convert the notebook to markdown
    body, resources = exporter.from_notebook_node(notebook)

    # Combine frontmatter with notebook markdown
    combined_markdown = f"{frontmatter_content}\n\n{body}"

    # Generate the output file path
    notebook_name = os.path.splitext(os.path.basename(notebook_path))[0]
    markdown_file = os.path.join(markdown_dir, f"{notebook_name}.md")

    # Write the combined markdown to a file
    with open(markdown_file, "w", encoding="utf-8") as f:
        f.write(combined_markdown)

    Logger.success_conversion(notebook_path, markdown_file, "with frontmatter")
