# Text Segmentation

A simple pre-project in python with the handwritten text segmentation module in c++.

### Requirements

* GCC/G++ 8+
* Python 3.7
* openCV 3+

### Run

``
python main.py -c -p
``

or 

``
python3 main.py -c -p
``


Specify an image

``
python main.py -c -p --image xxx.png
``

or

``
python3 main.py -c -p --image xxx.png
``

## Techniques

* Document Scanner
* Binarization with illumination compensation
* Line Segmentation with deslanting
* Word Segmentation

### Document Scanner

Process of detecting the predominant contour in the image and segment using a four-point transformation. [[ref]](https://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/)

### Binarization

A technique for light compensation and sauvola binarization was applied, but others techniques was studied also.

* Implementation of the paper "Efficient Illumination Compensation Techniques for text images", Guillaume Lazzara and Thierry Géraud, 2014. [[ref]](https://github.com/fanyirobin/text-image-binarization)
* Niblack, Sauvola and Wolf binarizations. [[ref]](https://github.com/chriswolfvision/local_adaptive_binarization)

### Line Segmentation

* Implementation of the paper "A Statistical approach to line segmentation in handwritten documents", Manivannan Arivazhagan, Harish Srinivasan and Sargur Srihari, 2007. [[ref]](https://github.com/Samir55/Image2Lines)

* Deslanting image. [[ref]](https://github.com/githubharald/DeslantImg)

### Word Segmentation

* Implementation of the paper "Scale Space Technique for Word Segmentation in Handwritten Documents", R. Manmatha and N. Srimal, 1999. [[ref]](https://github.com/githubharald/WordSegmentation)

**Binary image**

<img src="https://github.com/arthurflor/handwritten-text-segmentation/blob/master/doc/results/003.png/003_2_binary.png" width="680">

**Image lines**

<img src="https://github.com/arthurflor/handwritten-text-segmentation/blob/master/doc/results/003.png/003_3_lines.png" width="680">

**First line/words segment**

<img src="https://github.com/arthurflor/handwritten-text-segmentation/blob/master/doc/results/003.png/003_4_summary_001.png" width="680">


