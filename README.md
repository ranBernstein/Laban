# Laban Movement Analysis

The structure of this repository is:
- Laban: Recording and analysis code.
- Laban Paper: Latex and related files for building the paper "Multitask Learning for Laban Movement Analysis".

## Recordings and Annotations

The recordings from the paper are contained in the directory `/Laban/LabanLib/recordings`.
A record off all the anotations is contained in the file `Laban/Laban/LabanLib/LabanUtils/combination.txt`.
The entries are of the form `motor_element_1, motor_element_2, motor_element_3 folder_name_and_recording_number`.
For example, `ArmsToUpperBody,  Bind,  Sink s15` means that in every CMA's
recording folder (note that not every folder is a CMA folder), under the folder `sad`
the recordings that start with 15 in their file name include `ArmsToUpperBody`, `Bind` and `Sink` motor elements.
