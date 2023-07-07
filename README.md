# Code and documentation
This repository contains the code used in Rietze et al. (*in prep*): Summer drought weakens land surface cooling by tundra vegetation.

The necessary data is publicly available under [![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.7886426-blue)](https://doi.org/10.5281/zenodo.7886426).

## Repository structure:
There are three code folders that serve for the different processing steps in this study. Here is the structure and the included files in each folder:

```bash
├───code
│   │   environment.yml
│   │
│   ├───analysis
│   │   │   3_analysis.py
│   │   │   4_spei.R
│   │   │   FigFunctions.py
│   │   
│   ├───classification
│   │   │   0_resampling.py
│   │   │   1_1_prep_classification.py
│   │   │   1_1_runGLCM.R
│   │   │   1_classification.py
│   │   │   modules.py
│   │
│   └───thermal_drift_correction
│           main.py
│           modules.py
│           pydrone.yml
│
└───data
```

- The scripts in `analysis` are used to generate the main and supplementary figures as well as the supporting tables.
- The scripts in `classification` are used to prepare the drone imagery and run the random forest classification.
- The scripts in `thermal_drift_correction` are used to remove the temperature drift for all images in a thermal flight.
- The folder `data` is empty and should contain the data that can be downloaded from Zenodo (see link on top).

[to top](https://github.com/nrietze/ArcticThermoregulation/main/README.md)

## Software requirements
The data pre-processing and data analysis was using Python 3.10.6, pandas (0.24.2), GDAL 3.5.2 and R 4.2.2. Newer versions of these software packages will likely work, but have not been tested. You can find the conda environments as `.yml` files in this repository. The file `drone.yml` can be used to build the environment neccessary for the drift correction. The file `environment.yml`can be used to build a conda environment for the scripts used in the classification and analysis.

Code development and processing were carried out in Windows 10 (64 bit), but execution should (in theory) be platform independent.

[to top](https://github.com/nrietze/ArcticThermoregulation/main/README.md)

## Contact
Code development and maintenance: Nils Rietze ([nils.rietze@uzh.ch](nils.rietze@uzh.ch))

[to top](https://github.com/nrietze/ArcticThermoregulation/main/README.md)

## Acknowledgements
From the manuscript:
*N.R. was supported through the TRISHNA Science and Electronics Contribution (T-SEC), an ESA PRODEX project (contract no. 4000133711). Drone data acquisition was supported by the University Research Priority Program on Global Change and Biodiversity of the University of Zurich and by the Swiss National Science Foundation (grant no. 178753). We would like to thank Geert Hensgens of VU Amsterdam for sharing the flux tower data at the research site with us.*

[to top](https://github.com/nrietze/ArcticThermoregulation/main/README.md)
<!--- ## License
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons Licence" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
-->

## Citation
When citing elements in this repository, please cite as:

Rietze, Nils, Assmann, Jakob J., Damm, Alexander, Naegeli, Kathrin, Karsanaev, Sergey V., Maximov, Trofim C.,Plekhanova, Elena, Schaepman-Strub, Gabriela. *in prep*. Summer drought weakens land surface cooling by tundra vegetation. 

[to top](https://github.com/nrietze/ArcticThermoregulation/main/README.md)
