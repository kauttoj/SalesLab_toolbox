# saleslab_codes
Collection of codes to process and analyze iMotions datasets.

Main function is "SalesLab_ratareader.py" which reads and processes raw iMotions export datafiles. It will print preview images and separated datafiles for each modality it find in the data.

Requirements:

-Python 3.7+

-Numpy

-Pandas

-Scipy

-Ledapy 1.2+

-wxPython (only if using GUI option)

-iMotions software to collect data :)

Installation:

Probably easiest to install the latest Miniconda (docs.conda.io/en/latest/miniconda.html) and requires modules. Then unpack the codes into some folder and run the code (see the following).


There are two ways to use SalesLab_ratareader.py:

(1) Command line with input by calling "python SalesLab_ratareader.py c:\mypath\my_raw_input_file.txt", where you give a raw data file

(2) Via GUI by calling script without any inputs, i.e., "python SalesLab_ratareader.py", then you can drag and drop the file to process


What happens during analysis:

(1) Check that raw data fullfills the expected format and contains relevant information (GSR, eyetracking, heartrate and/or Affdex)

(2) Splitting raw data according to their data sources and removing temporally overlapped events (by taking median).

(3) Removal of bad data-points and doing additional computations (e.g., deconvolution of GSR signal)

(4) Printing results as tab-separated text files and preview plots for each data source. If there are (manual) annotations, they are automatically included for all sources as extra data.

(5) Preprocessing of GSR signal into phasic (fast) and tonic (slow) components using Ledalab continuous decomposition algorithm.

Note: We assume certain hard-coded properties in the data, such as column and event naming. If these change at some point, e.g., by iMotions software update, the pipeline might fail and needs updating.

Janne Kauttonen
3.10.2019 (initial release)
