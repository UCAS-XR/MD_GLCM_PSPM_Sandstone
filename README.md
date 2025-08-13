# MD_GLCM_PSPM_Sandstone

Code for "Multi-distance GLCM texture characteristics and particle size prediction model of sandstone".

What is this repository for?
This code is used to extract texture characteristics from color images of sandstone surfaces and predict their median grain size.

How do I get set up?
Copy the repository folder (Downloads from the menu on the left of this page) in Windows 64bit. In alternative use any other machine with python3 installed. If using the non-compiled python script, satisfy the dependencies listed.

Usage
1) First, run Center_corp.py to crop all color/gray-scale images to a uniform size of 3072*3072 pixels;
2) Next, run MD_GLCM_PSPM_Sandstone.py to preprocess the cropped images and obtain the results of the GLCM analysis. The image processing includes gray-scale conversion, extreme reflection removal, linear stretching, and gray-scale compression, while the GLCM analysis results include color images of the GLCM matrix and the data of GLCM feature values in tabular form；
3) Finally, run Multi-distance_GLCM.py to obtain the whole GLCM matrix for all selected distances. Each matrix is the arithmetic mean of all selected directions (optional).

How do I perform a quick test of code availability?
In addition to the above three code files, this repository also provides a quick test file (Quick_test.py) and a test image (test_image.bmp). The specific test steps are as follows:
1) First, run the first part of the test on the 11th line of the code (with the comment "Test_1" at the end of the line). This part is used to test the main function of Center_corp.py, which is to verify that the size of the cropped image is 3072*3072 pixels. If this part of the test is successful, the message "✅✅ Test_1 verification passed!" will be displayed；
2) Next, run the second part of the test (including Test_2.1-Test_2.7) on the 54th line of the code (with the comment "Test_2" at the end of the line). This part is used to test the main functions of MD_GLCM_PSPM_Sandstone.py, including the quantity, format, and quality of the test data. If this part of the test is successful, the message "✅✅ Test_2 (Test_2.1-Test_2.7) verification passed!" will be displayed；
3) Finally, run the final section (Section 3) of the test on line 195 of the code (with the comment "Test_3" at the end of the line). This section is designed to test the primary functionality of Multi-distance_GLCM.py, primarily verifying the dimensions of the whole GLCM matrix and the number of stored files. If the final section is successful, the message "✅✅ Test_3 verification passed!" will be displayed.
