import os
import shutil
import glob
import xml.etree.ElementTree as ET

BASE_DIR = "/home/tfpxxray/Desktop/Ph2_ACF/Ph2_ACF/module_testing"  # Directory where XML, TXT files, and module folders are stored

def modify_xml(xml_file, module_name, module_type, voltage_trim_values):
    "Modify the XML file to update the module name, configFile, and voltage trim values."
    try:
        # Check the XML file
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Determine the v1 or v2 type
        rd53_tag = f"RD53B{module_type}"  # RD53Bv1 or RD53Bv2

        # Update the Hybrid name and configFile
        for hybrid in root.findall(".//Hybrid"):
            hybrid.set("Name", module_name)
            for rd53b in hybrid.findall(rd53_tag):
                chip_id = rd53b.get("Id")  # Retrieve the Chip ID
                old_config = rd53b.get("configFile")
                if old_config:
                    # Update the config file to point to the OUT.txt file with this chip ID
                    # Format: CMSIT_RD53_{module_name}_0_{chip_id}_OUT.txt
                    new_config = f"CMSIT_RD53_{module_name}_0_{chip_id}_OUT.txt"
                    rd53b.set("configFile", new_config)

                # Update VOLTAGE_TRIM_DIG and VOLTAGE_TRIM_ANA
                settings = rd53b.find("Settings")
                if settings is not None and chip_id in voltage_trim_values:
                    trim_values = voltage_trim_values[chip_id]
                    settings.set("VOLTAGE_TRIM_DIG", trim_values["DIG"])
                    settings.set("VOLTAGE_TRIM_ANA", trim_values["ANA"])

        # Write the changes back to the file
        tree.write(xml_file, encoding="utf-8", xml_declaration=True)
        print(f"Modified XML file: {xml_file}")
    except Exception as e:
        print_error(f"Error modifying XML file {xml_file}: {e}")

# Errors will be in red
RESET_COLOR = "\033[0m"  # Reset to default terminal color
RED_COLOR = "\033[91m"   # Red color

def print_error(message):
    print(f"{RED_COLOR}Error: {message}{RESET_COLOR}")


def main():
    # Get module name and thermal cycle number
    module_name = input("Enter the module name: ").strip()
    thermal_cycle = input("Enter the thermal cycle number: ").strip()
    
    # Define module's directory
    module_dir = os.path.join(BASE_DIR, module_name, "xray")
    if not os.path.exists(module_dir):
        print(f"Module directory {module_dir} does not exist. Creating it...")
        os.makedirs(module_dir, exist_ok=True)
        print(f"Created module directory: {module_dir}")

    # Define the new thermal cycle folder
    thermal_cycle_dir = os.path.join(module_dir, f"ThermalCycle_{thermal_cycle}")
    os.makedirs(thermal_cycle_dir, exist_ok=True)
    print(f"Created thermal cycle directory: {thermal_cycle_dir}")

    # Define module type (v1 or v2)
    module_type = input("Enter the module type (v1 or v2): ").strip().lower()
    if module_type not in ["v1", "v2"]:
        print_error("Error: Invalid module type. Please enter 'v1' or 'v2'.")
        return

    # Define chip type (dual or quad)
    chip_type = input("Enter the chip type (dual or quad): ").strip().lower()
    if chip_type not in ["dual", "quad"]:
        print_error("Error: Invalid chip type. Please enter 'dual' or 'quad'.")
        return

    # Select the correct XML file based on module type and chip type
    xml_file = os.path.join(BASE_DIR + "/xray_files", f"CMSIT_xray_noise_CROC{module_type}_{chip_type}.xml")
    if not os.path.exists(xml_file):
        print_error(f"Error: XML file {xml_file} does not exist!")
        return

    # Copy the XML file to the thermal cycle folder
    new_xml_file = os.path.join(thermal_cycle_dir, os.path.basename(xml_file))
    try:
        shutil.copy(xml_file, thermal_cycle_dir)
        print(f"Copied XML file {xml_file} to {thermal_cycle_dir}")
    except Exception as e:
        print_error(f"Error copying XML file: {e}")
        return

    # Retrieve chip IDs from XML
    tree = ET.parse(new_xml_file)
    root = tree.getroot()
    rd53_tag = f"RD53B{module_type}"
    chip_ids = [rd53b.get("Id") for rd53b in root.findall(f".//{rd53_tag}")]

    # Ask user for VOLTAGE_TRIM_DIG and VOLTAGE_TRIM_ANA for each chip
    voltage_trim_values = {}
    for chip_id in chip_ids:
        print(f"Enter values for Chip ID {chip_id}:")
        dig_trim = input("  VOLTAGE_TRIM_DIG: ").strip()
        ana_trim = input("  VOLTAGE_TRIM_ANA: ").strip()
        voltage_trim_values[chip_id] = {"DIG": dig_trim, "ANA": ana_trim}

    # Modify the copied XML file
    modify_xml(new_xml_file, module_name, module_type, voltage_trim_values)

    # Define the pattern for tuned TXT files - updated to match original OUT.txt filenames
    tuned_txt_pattern = os.path.join(BASE_DIR, "xray_files/tuned_txt_files", module_name, f"CMSIT_RD53_{module_name}_0_*_OUT.txt")

    # Find all TXT files matching the pattern
    txt_files = glob.glob(tuned_txt_pattern)
    if not txt_files:
        print_error(f"Error: No TXT files matching pattern {tuned_txt_pattern} were found!")
        return

    # Copy all matched TXT files to the thermal cycle folder
    try:
        for txt_file in txt_files:
            shutil.copy(txt_file, thermal_cycle_dir)
            print(f"Copied TXT file {os.path.basename(txt_file)} from {tuned_txt_pattern} to {thermal_cycle_dir}")
    except Exception as e:
        print_error(f"Error copying TXT files: {e}")
        return

    # Inform the user that setup is complete
    print(f"Setup complete. Noise scan can now be run in the directory: {thermal_cycle_dir}")

if __name__ == "__main__":
    main()