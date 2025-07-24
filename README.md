# MD_GLCM_PSPM_Sandstone
Code for "Multi-distance GLCM texture characteristics and particle size prediction model of sandstone"

This code is developed on Python 3.11 and includes Center_corp.py, MD_GLCM_PSPM_Sandstone.py, and Multi-distance_GLCM.py.  

Specifically, the Center_corp.py file is used to crop an image of any size to a 3072x3072-pixel image from its center;

the MD_GLCM_PSPM_Sandstone.py is the main part of the code, including image preprocessing (such as gray-scale conversion, extreme reflection removal, linear stretching of dynamic range, and gray-scale compression), GLCM matrix calculation and visualization, and feature value calculation;  

the Multi-distance_GLCM.py is used to calculate the arithmetic mean of the GLCM matrix in the 0°, 45°, 90°, and 135° directions. 
