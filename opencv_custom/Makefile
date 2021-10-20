selectivesearchsegmentation_opencv_custom.so:
	g++ $(basename $@).cpp -o $@ -shared -fPIC -lopencv_core -lopencv_imgproc -I$(OPENCVINCLUDEDIR) -L$(OPENCVLIBDIR)

.PHONY: clean
clean:
	rm selectivesearchsegmentation_opencv_custom.so
