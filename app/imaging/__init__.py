"""CT and MRI imaging: import DICOM series, de-identify them, run trained
models slice by slice with abstention and region heatmaps, and let a
radiologist sign a report that's exported as a DICOM Structured Report.

- dicom.py: reading files and zips, series assembly, pixels and windows
- deid.py: DICOM de-identification (PS3.15 Basic Application Confidentiality Profile)
- store.py: studies, series, analyses and reports in the database and on disk
- analysis.py: the imaging model interface and the training-library adapter
- report.py: DICOM Structured Report output
- dicomweb.py: receiving (STOW-RS) and pulling from a PACS (QIDO-RS, WADO-RS)

Research and evaluation only: nothing here is a medical device."""
