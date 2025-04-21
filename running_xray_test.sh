#!/bin/bash

# Paths
BASE_DIR="/home/tfpxxray/Desktop/Ph2_ACF/Ph2_ACF"
MODULE_TESTING_DIR="${BASE_DIR}/module_testing"
PYTHON_SCRIPT="xray_test.py"
PANTHERA_SCRIPT="panthera_downloader.py"

# Use the full path to run_xray_analysis.py
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
XRAY_ANALYSIS_SCRIPT="${SCRIPT_DIR}/run_xray_analysis.py"
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
RESET="\033[0m"

echo -e "${GREEN}=== Starting the Ph2 ACF Automation Script ===${RESET}"

# Step 1: Prompt the user for module name, thermal cycle, and chip type
echo -e "${GREEN}Please provide the required module information. This is for PH2 ACF commands${RESET}"
read -p "Enter the module name: " MODULE_NAME
read -p "Enter the thermal cycle number: " THERMAL_CYCLE
read -p "Enter the chip type (dual or quad): " CHIP_TYPE

# Validate chip type
while [[ ! "$CHIP_TYPE" =~ ^(dual|quad)$ ]]; do
    echo -e "${RED}Invalid chip type. Please enter 'dual' or 'quad'.${RESET}"
    read -p "Enter the chip type (dual or quad): " CHIP_TYPE
done

# For quad modules, ask for chip version
CHIP_VERSION=""
if [[ "$CHIP_TYPE" == "quad" ]]; then
    read -p "Enter the chip version (v1 or v2): " CHIP_VERSION
    while [[ ! "$CHIP_VERSION" =~ ^(v1|v2)$ ]]; do
        echo -e "${RED}Invalid chip version. Please enter 'v1' or 'v2'.${RESET}"
        read -p "Enter the chip version (v1 or v2): " CHIP_VERSION
    done
fi

# Set bias voltage to 80V (always)
BIAS_VOLTAGE=80

echo "Module Name: $MODULE_NAME"
echo "Thermal Cycle Number: $THERMAL_CYCLE"
echo "Chip Type: $CHIP_TYPE"
if [[ -n "$CHIP_VERSION" ]]; then
    echo "Chip Version: $CHIP_VERSION"
fi
echo "Bias Voltage: $BIAS_VOLTAGE V"

# Step 1.5: Ask if the user wants to copy files from Downloads
read -p "Do you want to copy calibration files from the Downloads folder? (y/n): " COPY_FROM_DOWNLOADS
if [[ "$COPY_FROM_DOWNLOADS" =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}=== Copying files from Downloads folder ===${RESET}"
    
    # Check if panthera_downloader.py exists
    if [ ! -f "$PANTHERA_SCRIPT" ]; then
        echo -e "${RED}Error: File copy script $PANTHERA_SCRIPT not found!${RESET}"
        echo -e "${YELLOW}Continuing without copying files from Downloads...${RESET}"
    else
        # Run the file copy script with the module name and chip type as arguments
        python "$PANTHERA_SCRIPT" --module "$MODULE_NAME" --chip-type "$CHIP_TYPE" || { 
            echo -e "${RED}Error: Copying files from Downloads failed!${RESET}"
            
            read -p "Do you want to continue without copying files? (y/n): " CONTINUE_WITHOUT_COPY
            if [[ ! "$CONTINUE_WITHOUT_COPY" =~ ^[Yy]$ ]]; then
                echo "Exiting..."
                exit 1
            fi
        }
        echo -e "${GREEN}Files copied successfully.${RESET}"
    fi
fi

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

# Modify GTX RX polarity in XML file based on module name
if [[ "$CHIP_TYPE" == "quad" && "$CHIP_VERSION" == "v2" ]]; then
    echo -e "${GREEN}Checking if GTX RX polarity needs to be modified...${RESET}"
    
    # Set gtx_rx_polarity based on module name
    if [[ "$MODULE_NAME" == SH01* ]]; then
        echo -e "${GREEN}Module $MODULE_NAME starts with SH01, setting gtx_rx_polarity fmc_l12 to 0b1001${RESET}"
        NEW_POLARITY="0b1001"
    elif [[ "$MODULE_NAME" == SH00* ]]; then
        echo -e "${GREEN}Module $MODULE_NAME starts with SH00, setting gtx_rx_polarity fmc_l12 to 0b1101${RESET}"
        NEW_POLARITY="0b1101"
    else
        echo -e "${YELLOW}Module $MODULE_NAME does not start with SH00 or SH01. Not modifying XML.${RESET}"
    fi
    
    # Update the XML file if NEW_POLARITY is set
    if [[ -n "$NEW_POLARITY" ]]; then
        # Create a backup of the original file
        cp "$XML_FILE" "${XML_FILE}.bak"
        
        # Use sed to replace the gtx_rx_polarity value
        sed -i "s/<Register name=\"fmc_l12\">0b[01]\{4\}<\/Register>/<Register name=\"fmc_l12\">$NEW_POLARITY<\/Register>/g" "$XML_FILE"
        
        echo -e "${GREEN}Updated gtx_rx_polarity in $XML_FILE${RESET}"
    fi
else
    echo -e "${YELLOW}Not a quad v2 module. No need to modify GTX RX polarity.${RESET}"
fi

# Step 4: Run Ph2 ACF commands
echo -e "${GREEN}=== Running Ph2 ACF Commands ===${RESET}"

echo "1. Running fpgaconfig to upload the firmware..."
fpgaconfig -c "$XML_FILE" -i QUAD_ELE_CROC_v5-0.bit || { echo "Error: fpgaconfig failed"; exit 1; }
echo "Firmware uploaded successfully."

echo "2. Running CMSITminiDAQ for resetting..."
CMSITminiDAQ -f "$XML_FILE" -r || { echo "Error: CMSITminiDAQ reset failed"; exit 1; }
echo "CMSITminiDAQ reset completed successfully."

echo "3. Running CMSITminiDAQ for noise scan..."
CMSITminiDAQ -f "$XML_FILE" -c noise || { echo "Error: CMSITminiDAQ noise scan failed"; exit 1; }
echo "CMSITminiDAQ noise scan completed successfully."

echo -e "${GREEN}=== All commands executed successfully! === ${RESET}"

# Step 5: Ask if the user wants to run X-ray analysis
read -p "Do you want to run the X-ray analysis now? (y/n): " RUN_XRAY_ANALYSIS
if [[ "$RUN_XRAY_ANALYSIS" =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}=== Running X-ray Analysis ===${RESET}"
    
    # Check if the analysis script exists
    if [ ! -f "$XRAY_ANALYSIS_SCRIPT" ]; then
        echo -e "${RED}Error: X-ray analysis script $XRAY_ANALYSIS_SCRIPT not found!${RESET}"
        echo -e "${YELLOW}Skipping X-ray analysis...${RESET}"
    else
        # Navigate back to the script directory
        cd "$SCRIPT_DIR" || { echo "Error: Failed to navigate to script directory"; exit 1; }
        
        # Run the analysis script
        python "$XRAY_ANALYSIS_SCRIPT" \
            --module "$MODULE_NAME" \
            --thermal-cycle "$THERMAL_CYCLE" \
            --chip-type "$CHIP_TYPE" \
            --bias "$BIAS_VOLTAGE" || {
            echo -e "${RED}Error: X-ray analysis failed!${RESET}"
        }
    fi
else
    echo -e "${YELLOW}Skipping X-ray analysis. You can run it later using:${RESET}"
    echo -e "${YELLOW}python $XRAY_ANALYSIS_SCRIPT --module $MODULE_NAME --thermal-cycle $THERMAL_CYCLE --chip-type $CHIP_TYPE --bias $BIAS_VOLTAGE${RESET}"
    echo -e "${GREEN}Note: The analysis script will automatically find the latest SCurve.root file in your Downloads folder.${RESET}"
fi

echo -e "${GREEN}=== Workflow completed! === ${RESET}"