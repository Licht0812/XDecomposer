---
slug: "/tutorial"
name: "tutorial"
hasSider: false
title: "Tutorial"
---

# Dara Server Tutorial

<p align="center">
<img src="images/dara_explained.png" alt="Dara Explained" width="70%">
</p>
<p align="center" style="font-size:1.05em; color: #3a3a3a">
Dara is designed for <strong>automated phase identification and refinement</strong> for powder XRD data.
</p>

## Overview

The Dara server provides a web-based interface for automated phase identification from powder X-ray diffraction (XRD) data.

## Getting Started
You'll see the main interface with navigation options:

- **Submit**: Upload and analyze new XRD patterns
- **Results**: Browse all submitted analyses
- **Tutorial**: A short tutorial as you are reading.
- **Documentation**: The documentation for the Dara package.

## Submitting an Analysis

### Step 1: Navigate to Submit Page

Click on "Submit" in the navigation menu to access the submission form.

### Step 2: Upload Your XRD Pattern

**Supported File Formats:**
- `.xy` - Two-column ASCII format (2θ, intensity)
- `.txt` - Text files with diffraction data
- `.xye` - Three-column format (2θ, intensity, error)
- `.xrdml` - PANalytical/Malvern Panalytical XML format
- `.raw` - Rigaku/Bruker raw data format
- `.rasx` - Rigaku RASX powder diffraction format

### Step 3: Specify Precursor Information

**Precursor Formulas:**
Enter the chemical formulas of your starting materials, splitted by comma.
```
CaO, TiO2
```

or you can write the target phase formula in the text box, like
```
CaTiO3
```

Dara only needs to know what elements are needed to include for the search.

### Step 4: Configure Analysis Parameters

**User Identification:**
Enter your username for tracking and organization purposes. You can use this option to better organize your patterns.

**Instrument Profile:**
Select your diffractometer configuration from the dropdown.

**Reaction Predictor (Optional):**
- **Enable**: Check this box to use reaction prediction
- **Temperature**: Specify reaction temperature in Celsius (-273°C minimum)
- **Disable**: Leave unchecked to perform standard database search

### Step 5: Submit Analysis

Click "Submit Analysis" to queue your job. You will see if there is any error in the submission. Otherwise, you will see the task submitted successfully and appear in the results page.

## Monitoring Your Analysis

### Results Page

Navigate to "Results" to see all submitted analyses:

**Table Columns:**
- **ID #**: Unique task identifier
- **Name**: Original filename
- **Status**: Current processing state
  - 🔵 `PENDING`: Waiting in queue
  - 🟡 `RUNNING`: Currently processing
  - 🟢 `COMPLETED`: Analysis finished successfully
  - 🔴 `FIZZLED`: Analysis failed with error
- **Created**: Submission timestamp
- **Submitted by**: Username

**Filtering:**
Use the search box to filter results by username.

**Pagination:**
Navigate through results using the pagination controls at the bottom.

## Interpreting Results

### Task Details Page

Click on any completed analysis to view detailed results:

### Status Information
- **Task Label**: Original filename
- **Status**: Final processing state
- **Runtime**: Total computation time
- **Timestamps**: Submission, start, and completion times

### Phase Identification Results

**Best Fit (Rwp):**
A numerical value indicating fit quality (lower numbers indicate better fits).

**Identified Phases:**
- **Representative Phases**: Primary phase in each compositional group
- **Phase Groups**: Related phases grouped by composition
- **Highlighted Phases**: Phases significantly contributing to the pattern

**Phase Information:**
- Chemical formula with proper subscripts
- Structural identification from COD database
- Relative abundance in refinement

### Visualization

**Refinement Plots:**
Interactive plots are displayed showing your experimental data overlaid with the calculated fit.

**Plot Controls:**
- Use your mouse to zoom, pan, and hover for detailed inspection
- Click buttons to switch between different refinement results
- Right-click on plots to download as images

### Multiple Solutions

The analysis typically provides several possible solutions ranked by fit quality:

1. **Best Solution**: Lowest Rwp value
2. **Alternative Solutions**: Other plausible phase combinations
3. **Comparison**: Easy switching between different interpretations

## Advanced Features

## API Access

For programmatic access, the server provides RESTful API endpoints:

### Submit Analysis
```bash
POST /api/submit
```

You can use this Python code to submit an analysis:
```python
import requests
from requests.auth import HTTPBasicAuth
import re
from pathlib import Path

user = "your_username"
base_url = "your dara server url"  # e.g. http://localhost:8898

def submit(user, file_path, precursor_or_element, use_reaction_network=False, temperature_C=-1000):
    url = f"{base_url}/api/submit"
    files = {'pattern_file': open(file_path, 'rb')}
    data = {
        'user': user,
        'use_rxn_predictor': use_reaction_network,
        'temperature': temperature_C,
        'precursor_formulas': str(precursor_or_element)
    }
    # Note: Replace with your actual authentication if needed
    response = requests.post(
        url,
        files=files,
        data=data,
    )
    print(f"Submitted: {file_path} with data: {data}")
    return response.json()

def parse_phases(filename):
    # Extract phase names from filename, e.g. "BaTiO3-SrTiO3_xxx.xrdml" -> ["BaTiO3", "SrTiO3"]
    return [re.sub(r'^\d+', '', p) for p in filename.split("_")[0].split("-")]

# Example usage:
dataset_path = Path("/path/to/your/xrdml/files")

for file in dataset_path.glob("*.xrdml"):
    phases = parse_phases(file.name)
    submit(user, file, phases)


### Get Task Status
```bash
GET /api/task/{task_id}
```


## Troubleshooting

### Common Issues

**File Upload Fails:**
- Check file format is supported
- Ensure file is not corrupted and correctly formatted

**Poor Results:**
- Try different instrument profiles from the dropdown
- Double-check your wavelength setting
- Verify your precursor formulas are entered correctly
