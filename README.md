# This is an implementation of the paper "Learning Double-Compression Video Fingerprints Left From Social-Media Platforms".

## Installation
    $ git clone https://github.com//stervt/Learning-Double-Compression-Video-Fingerprints-Left-From-Social-Media-Platforms
    $ cd Learning-Double-Compression-Video-Fingerprints-Left-From-Social-Media-Platforms/
    $ sudo pip3 install -r requirements.txt

## Implementation   
### LEARNING DOUBLE-COMPRESSION VIDEO FINGERPRINTS LEFT FROM SOCIAL-MEDIA PLATFORMS
__

#### Authors
Irene Amerini, Aris Anagnostopoulos, Luca Maiano, Lorenzo Ricciardi Celsi
#### Abstract
Social media and messaging apps have become major communication platforms. Multimedia contents promote improved user engagement and have thus become a very important communication tool. However, fake news and manipulated content can easily go viral, so, being able to verify the source of videos and images as well as to distinguish between native and downloaded content becomes essential. Most of the work performed so far on social media provenance has concentrated on images; in this paper, we propose a CNN architecture that analyzes video content to trace videos back to their social network of origin. The experiments demonstrate that stating platform provenance is possible for videos as well as images with very good accuracy

[[Paper]](https://ieeexplore.ieee.org/document/9413366) 

## Useage
* This papaer used [VISION](https://lesc.dinfo.unifi.it/en/datasets) dataset, so if you want to use the whole dataset, please use wget to download the VISION dataset. Then run `get_specified_frame.ipynb` and `create_dataset.ipynb` in your jupter notebook to crete dataset.

* If you want to run for the sample case, just run `main.ipynb` in your jupyter notebook.

NOTE: The paper just used a subset of VISION datase and the author did not specify which videos were used. Here the sample used the videos of devices from D01 to D20.

## References
[1] Irene Amerini, Aris Anagnostopoulos, Luca Maiano, Lorenzo Ricciardi Celsi. *Learning Double-Compression Video Fingerprints Left From Social-Media Platforms*.   