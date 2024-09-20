import re
from nbconvert.preprocessors import Preprocessor


class StripANSICodesPreprocessor(Preprocessor):
    ANSI_ESCAPE = re.compile(
        r"""
        \x1B  # ESC
        (?:   # 7-bit C1 Fe (except CSI)
            [@-Z\\-_]  # or [0-9A-ORZcf-ori]
        |     # or 8-bit C1 Fe
            \[[0-?]*[ -/]*[@-~]
        )""",
        re.VERBOSE,
    )

    def preprocess_cell(self, cell, resources, index):
        # If the cell contains outputs, clean up ANSI escape codes
        if "outputs" in cell:
            for output in cell["outputs"]:
                if "text" in output:
                    output["text"] = self.ANSI_ESCAPE.sub("", output["text"])
        return cell, resources


class HideWidgetOutputPreprocessor(Preprocessor):
    # Regex patterns to detect file upload widgets in the output
    widget_output_patterns = [
        re.compile(r'input type="file"'),  # Detect HTML input elements
        re.compile(r"Upload widget is only available"),  # Detect widget messages
    ]

    def preprocess_cell(self, cell, resources, index):
        # If the cell is a code cell and has outputs
        if cell.cell_type == "code" and "outputs" in cell:
            # Check each output for widget patterns
            cleaned_outputs = []
            for output in cell["outputs"]:
                if not self.contains_widget_output(output):
                    cleaned_outputs.append(output)
            # Replace the outputs with the cleaned ones
            cell["outputs"] = cleaned_outputs
        return cell, resources

    def contains_widget_output(self, output):
        # Check if the output contains any of the widget patterns
        if "text" in output:
            for pattern in self.widget_output_patterns:
                if pattern.search(output["text"]):
                    return True
        if "data" in output and "text/html" in output["data"]:
            for pattern in self.widget_output_patterns:
                if pattern.search(output["data"]["text/html"]):
                    return True
        return False


class HideLongPipInstallOutputPreprocessor(Preprocessor):
    # Regex pattern to detect pip install commands
    pip_install_pattern = re.compile(r"pip install")

    def preprocess_cell(self, cell, resources, index):
        # Only modify code cells
        if cell.cell_type == "code":
            if self.is_pip_install_cell(cell):
                # Check if there is an output and trim it
                cell["outputs"] = self.hide_long_output(cell["outputs"])
        return cell, resources

    def is_pip_install_cell(self, cell):
        # Check if the cell source contains a pip install command
        return self.pip_install_pattern.search(cell.get("source", ""))

    def hide_long_output(self, outputs):
        # Replace outputs that are too long with a placeholder message
        new_outputs = []
        for output in outputs:
            if "text" in output and len(output["text"]) > 500:
                output["text"] = "[Output too long, omitted for brevity]"
            new_outputs.append(output)
        return new_outputs
