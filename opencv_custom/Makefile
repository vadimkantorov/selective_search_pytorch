# default installation paths of [apt install libopencv-dev] on Ubuntu 20.04

OPENCVLIBDIR := /usr/lib/x86_64-linux-gnu
OPENCVINCLUDEDIR := /usr/include/opencv4

selectivesearchsegmentation_opencv_custom_.so:
	g++ -Ofast selectivesearchsegmentation_opencv_custom.cpp -o $@ -shared -fPIC -lopencv_core -lopencv_imgproc -I$(OPENCVINCLUDEDIR) -L$(OPENCVLIBDIR)

.PHONY: clean
clean:
	rm selectivesearchsegmentation_opencv_custom_.so
