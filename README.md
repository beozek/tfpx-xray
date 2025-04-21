
### Command-line Arguments

- **-m**: Module name
- **-t**: Thermal cycle number
- **-c**: Chip type (dual or quad)
- **-b**: Bias voltage (V)
- **-a**: Automatically find and run X-ray analysis

## Script Descriptions

### panthera_downloader.py

Copies the manually downloaded OUT.txt files:
- Looks for files matching the pattern "CMSIT_RD53_{module_name}_0_*_OUT.txt" in Downloads
- Copies them directly without renaming
- Places them in the correct directory for xray_test.py to use

### xray_test.py

Sets up module directories and configuration:
- Creates directories for the module and thermal cycle
- Copies and modifies the correct XML file based on module type and chip type
- Updates XML with module-specific information
- Copies calibrated TXT files to the thermal cycle directory

### running_xray_test.sh

Orchestrates the entire workflow:
- Runs the file copy script (optional)
- Executes the setup Python script
- Sources environment setup
- Runs FPGA configuration and CMSITminiDAQ commands
- Runs X-ray analysis (optional)

### run_xray_analysis.py

Performs X-ray analysis after the test:
- Finds and copies all necessary input files
- Uses xray_analysis.py to analyze the results
- Generates plots showing hits, noise, threshold, and open bumps
- Creates a list of open bump coordinates (open_bumps.txt)

# How to run manually:
# python xray_analysis.py -scurve Run0000041_SCurve -noise Run000000_NoiseScan -sensor SH0055

### xray_analysis.py

Core analysis script:
- Analyzes SCurve.root and NoiseScan.root files
- Identifies open bumps based on hit count and ToT thresholds
- Generates detailed plots for analysis
- Creates files listing open bump coordinates

## Troubleshooting

- **Missing TXT Files**: Make sure you've downloaded all required files from Panthera to the Downloads folder.
- **XML File Errors**: Make sure the module type, chip type, VDDD, VDDA values are specified correctly.
- **FPGA or CMSITminiDAQ Failures**: Check that the XML file is correctly formatted and all paths are valid.
- **Analysis Errors**: Make sure both SCurve.root and NoiseScan.root files are available.

## File Naming Conventions

- **Expected file format**: CMSIT_RD53_{module_name}_0_{chip_id}_OUT.txt
- **XML files**: CMSIT_xray_noise_CROC{v1/v2}_{dual/quad}.xml
