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

[to top](https://github.com/nrietze/ArcticThermoregulation/main/README.md)

## Contact
Code development and maintenance: Nils Rietze (nils.rietze@uzh.ch)

[to top](https://github.com/nrietze/ArcticThermoregulation/main/README.md)

## Acknowledgements
From the manuscript:
*N.R. was supported through the TRISHNA Science and Electronics Contribution (T-SEC), an ESA PRODEX project (contract no. 4000133711). Drone data acquisition was supported by the University Research Priority Program on Global Change and Biodiversity of the University of Zurich and by the Swiss National Science Foundation (grant no. 178753). We would like to thank Geert Hensgens of VU Amsterdam for sharing the flux tower data at the research site with us.*

[to top](https://github.com/nrietze/ArcticThermoregulation/main/README.md)

## Citation
When citing elements in this repository, please cite as:

Rietze, Nils, Assmann, Jakob J., Damm, Alexander, Naegeli, Kathrin, Karsanaev, Sergey V., Maximov, Trofim C.,Plekhanova, Elena, Schaepman-Strub, Gabriela. *in prep*. Summer drought weakens land surface cooling by tundra vegetation. 
[to top](https://github.com/nrietze/ArcticThermoregulation/main/README.md)
