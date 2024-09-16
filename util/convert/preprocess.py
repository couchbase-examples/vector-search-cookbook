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
