class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    OKGREENH = "\x1b[1;33;42m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    FAILH = "\x1b[1;33;41m"
    ENDC = "\033[0m"
    ENDH = "\x1b[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class Logger:
    @staticmethod
    def success(message, details=""):
        print(
            f"{bcolors.OKGREENH}  OK  {bcolors.ENDH} - {message}\n\t\t{bcolors.HEADER}{details}{bcolors.ENDC}"
        )

    @staticmethod
    def success_conversion(from_file, to_file, details=""):
        print(
            f"{bcolors.OKGREENH}  OK  {bcolors.ENDH} - Converted \n\t\t{bcolors.HEADER}{from_file}{bcolors.ENDC} to \n\t\t{bcolors.HEADER}{to_file}{bcolors.ENDC} \n\t {details}."
        )

    @staticmethod
    def fail(message, reason=""):
        print(
            f"{bcolors.FAILH} FAIL {bcolors.ENDH} - {bcolors.FAIL}{message} \n\t Reason: {reason}{bcolors.ENDC}"
        )

    @staticmethod
    def fail_conversion(fail_file, reason=""):
        print(
            f"{bcolors.FAILH} FAIL {bcolors.ENDH} - {bcolors.FAIL}Skipping conversion for {fail_file} \n\t Reason: {reason}.{bcolors.ENDC}"
        )
