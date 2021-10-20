selectivesearchsegmentation.so:
	g++ main.cpp -o $@ -shared -fPIC -lopencv_core -lopencv_imgproc -L../../prefix/miniconda/lib/ -I../../prefix/miniconda/include/opencv4/

.PHONY: clean
clean:
	rm selectivesearchsegmentation.so
