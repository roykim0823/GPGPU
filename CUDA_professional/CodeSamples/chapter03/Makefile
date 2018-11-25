#CU_APPS=nestedHelloWorld nestedReduce nestedReduce2 nestedReduceNosync \
#	    reduceInteger simpleDeviceQuery simpleDivergence sumMatrix
C_APPS=reduceInteger simpleDeviceQuery simpleDivergence sumMatrix

all: ${C_APPS} ${CU_APPS}

%: %.cu
	nvcc -O2 -arch=sm_30 -o $@ $< -lcudadevrt --relocatable-device-code true
%: %.c
	gcc -O2 -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
