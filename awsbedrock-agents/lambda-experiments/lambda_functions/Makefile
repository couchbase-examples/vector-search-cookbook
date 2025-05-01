# Makefile

# Specify the Python version (Note: pip --target doesn't depend on a specific venv Python version)
PYTHON_VERSION := python3.12 # This is less critical now, but kept for consistency if needed elsewhere
PACKAGE_DIR := package_dir # Temporary directory for packaging
PACKAGE_NAME := lambda_package.zip
# LAMBDA_FUNCTION_SOURCE_DIR variable removed for simplicity in install_deps, but concept remains

# Default target executed when no arguments are given to make.
default: clean package

# Install dependencies directly into the package directory
install_deps:
	mkdir -p $(PACKAGE_DIR)
	# Upgrade pip quietly
	python3 -m pip install --upgrade pip --quiet
	# Directly use the known relative path since make runs in the correct directory
	# Install dependencies targeting the Lambda environment (Amazon Linux 2, x86_64, Python 3.9)
	python3 -m pip install \
		-r requirements.txt \
		-t $(PACKAGE_DIR) \
		--platform manylinux2014_x86_64 \
		--python-version 3.9 \
		--implementation cp \
		--abi cp39 \
		--only-binary=:all: \
		--quiet

# Package the installed libraries and your lambda function into a zip
package: install_deps
	# Copy the lambda function script into the package directory
	cp lambda_function.py $(PACKAGE_DIR)
	# Zip the contents of the package directory
	cd $(PACKAGE_DIR); zip -r9 ../$(PACKAGE_NAME) .

# Clean up the environment
clean:
	rm -rf $(PACKAGE_DIR)
	rm -f $(PACKAGE_NAME)
	# Remove the old venv if it exists from previous runs
	rm -rf .lambda_venv

.PHONY: default install_deps package clean
