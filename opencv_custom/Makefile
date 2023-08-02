OPENCVLIBDIR := .
OPENCVINCLUDEDIR := .

selectivesearchsegmentation_opencv_custom_.so:
	g++ -Ofast selectivesearchsegmentation_opencv_custom.cpp -o $@ -shared -fPIC -lopencv_core -lopencv_imgproc -I$(OPENCVINCLUDEDIR) -L$(OPENCVLIBDIR)

.PHONY: clean
clean:
	rm selectivesearchsegmentation_opencv_custom_.so
