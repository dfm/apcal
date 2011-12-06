all: likelihood

clean:
	rm -f apcal/_likelihood.so

likelihood: apcal/_likelihood.c
	python setup.py build_ext --inplace
