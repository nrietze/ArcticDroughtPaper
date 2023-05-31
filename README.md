# ArcticThermoregulation
Code for the thermoregulation study.
- Nils Rietze - nils.rietze@uzh.ch

- Thermoregulaiton in the Siberian tundra

## Repo structure:
There are three code folders that serve for the different processing steps in this study. Here is the structure and the included files in each folder:

```bash
├───analysis
│   │   3_analysis.py
│   │   4_spei.R
│   │   FigFunctions.py
│
├───classification
│   │   0_resampling.py
│   │   1_1_prep_classification.py
│   │   1_1_runGLCM.R
│   │   1_classification.py
│   │   modules.py
│
└───thermal_drift_correction
        main.py
        modules.py
        pydrone.yml
```

- The scripts in `analysis` are used to generate the main and supplementary figures as well as the supporting tables.
- The scripts in `classification` are used to prepare the drone imagery and run the random forest classification.
- The scripts in `thermal_drift_correction` are used to remove the temperature drift for all images in a thermal flight.
