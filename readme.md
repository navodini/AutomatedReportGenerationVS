# **Clinical guideline driven report generation of vestibular schwannoma**

This is the official implementation of the study **'Artificial intelligence for personalised management of vestibular schwannoma: A multidisciplinary clinical implementation study'**.

Wijethilake, N. , Connor, S., Oviedova, A., Burger, R., De Leon De Sagun, J., Hitchings, A., ... & Shapey, J. (2023). Artificial intelligence for personalised management of vestibular schwannoma: A clinical implementation study within a multidisciplinary decision making environment. (Under review) https://doi.org/10.1101/2023.11.17.23298685

Wijethilake, N., Connor, S., Oviedova, A., Burger, R., Vercauteren, T., & Shapey, J. (2023). A Clinical Guideline Driven Automated Linear Feature Extraction for Vestibular Schwannoma (Accepted - SPIE Medical Imaging 2024) https://doi.org/10.48550/ARXIV.2310.19392

![model](outline.png)

### SampleCase

The directory contains a sample case with three MRI scans corresponding to three consecutive time points.

**Note:** For the clinical implementation study, we employed a two-stage deep learning model (based on (nnUNet)[https://github.com/MIC-DKFZ/nnUNet]) to generate the tumour masks. Here, we have provided manually annotated masks for each MRI scan.

### Usage

To use the repository, follow these general steps:

* **Feature Extraction:** Execute scripts in the FeatureExtraction directory to extract clinical guideline driven features from MRI masks.
* **Visualisation:** Execute scripts in the Visualisation directory to obtain visualisations of the axial views of the tumour with the delineated tumour.
* **Report Generation:** Execute scripts in the ReportGeneration directory to generate reports summarising the tumour behavior over time.


### Requirements

Refer to the requirements.txt file for the required Python packages.

## Feature Extraction

This directory contains all the supporting scripts that are used to obtain clinically used measurements. We have developed our scripts to extract the coordinates corresponding to the linear measurements, which can be used in the next step, i.e., visualisation.

* Run the FeatureExtraction/main.py script to generate all the measurements, which will be saved as JSON files.
* These features are specifically used in the radiological reporting of Vestibular Schwannoma. For further details, please refer to the following publication.
  * Connor, Steve EJ. "Imaging of the vestibular schwannoma: diagnosis, monitoring, and treatment planning." Neuroimaging Clinics 31.4 (2021): 451-471. https://doi.org/10.1016/j.nic.2021.05.006

## Visualisation

* Install SlicerJupyter using https://github.com/Slicer/SlicerJupyter for this step. 

* Set the following script parameter: 
  * **subjects_path**: Absolute path to the directory containing MRI data for different subjects.

* Run `Slicer_Visualisation.ipynb` to generate the cropped MRI axial view of the tumour.

## Report Generation

The ReportGeneration directory contains scripts for generating reports using the extracted features and the images generated in the visualisation step. Additionally, it includes an icons subdirectory with various image assets.

Run `ReportGeneration/main.py` the summarised reports with the linear and volume measurements for each time point. 

Feel free to explore specific scripts for more detailed information on each step of the pipeline.
