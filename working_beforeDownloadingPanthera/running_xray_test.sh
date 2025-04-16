#!/bin/bash

# Paths
BASE_DIR="/home/tfpxxray/Desktop/Ph2_ACF/Ph2_ACF"
MODULE_TESTING_DIR="${BASE_DIR}/module_testing"
PYTHON_SCRIPT="xray_test.py"
GREEN="\033[0;32m"
RED="\033[0;31m"
RESET="\033[0m"

echo -e "${GREEN}=== Starting the Ph2 ACF Automation Script ===${RESET}"

# Step 1: Prompt the user for module name and thermal cycle
echo -e "${GREEN}Please provide the required module information. This is for PH2 ACF commands${RESET}"
read -p "Enter the module name: " MODULE_NAME
read -p "Enter the thermal cycle number: " THERMAL_CYCLE
echo "Module Name: $MODULE_NAME"
echo "Thermal Cycle Number: $THERMAL_CYCLE"

# Step 2: Run the Python script
echo -e "${GREEN}=== Running the Python script ===${RESET}"
echo "Please provide the answers to set up module files and folders..."
python "$PYTHON_SCRIPT" || { echo "Error: Python script execution failed!"; exit 1; }
echo -e "${GREEN}Python script executed successfully.${RESET}"

# Construct the thermal cycle directory path
THERMAL_CYCLE_DIR="${MODULE_TESTING_DIR}/${MODULE_NAME}/xray/ThermalCycle_${THERMAL_CYCLE}"
echo "Constructed thermal cycle directory path: $THERMAL_CYCLE_DIR"

# Verify if the thermal cycle directory exists
if [ ! -d "$THERMAL_CYCLE_DIR" ]; then
    echo -e "${RED}Error: Thermal cycle directory $THERMAL_CYCLE_DIR does not exist!${RESET}"
    exit 1
fi
echo "Thermal cycle directory exists. Proceeding..."

# Step 3: Source setup.sh
echo -e "${GREEN}Sourcing setup.sh to configure the environment...${RESET}"
cd "$BASE_DIR" || { echo "Error: Failed to navigate to $BASE_DIR"; exit 1; }
if [ ! -f "setup.sh" ]; then
    echo "Error: setup.sh not found in $BASE_DIR!"
    exit 1
fi
source setup.sh
echo -e "${GREEN}Environment setup completed.${RESET}"

# Navigate to the thermal cycle directory
echo "Navigating to the thermal cycle directory: $THERMAL_CYCLE_DIR"
cd "$THERMAL_CYCLE_DIR" || { echo "Error: Failed to navigate to $THERMAL_CYCLE_DIR"; exit 1; }

# Verify XML file existence
echo -e "${GREEN}Looking for the XML file in the thermal cycle directory...${RESET}"
XML_FILE=$(ls CMSIT_xray_noise_CROC*.xml | head -n 1)
if [ -z "$XML_FILE" ]; then
    echo "Error: No XML file found in $THERMAL_CYCLE_DIR!"
    exit 1
fi
echo "XML file found: $XML_FILE"

# Step 4: Run Ph2 ACF commands
echo -e "${GREEN}=== Running Ph2 ACF Commands ===${RESET}"

echo "1. Running fpgaconfig to upload the firmware..."
fpgaconfig -c "$XML_FILE" -i QUAD_ELE_CROC_v4-9.bit || { echo "Error: fpgaconfig failed"; exit 1; }
echo "Firmware uploaded successfully."

echo "2. Running CMSITminiDAQ for resetting..."
CMSITminiDAQ -f "$XML_FILE" -r || { echo "Error: CMSITminiDAQ reset failed"; exit 1; }
echo "CMSITminiDAQ reset completed successfully."

echo "3. Running CMSITminiDAQ for noise scan..."
CMSITminiDAQ -f "$XML_FILE" -c noise || { echo "Error: CMSITminiDAQ noise scan failed"; exit 1; }
echo "CMSITminiDAQ noise scan completed successfully."

echo -e "${GREEN}=== All commands executed successfully! === ${RESET}"