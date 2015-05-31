########## the compilers and their flags #################

# the compilers:
CUDACC := nvcc

# flags:
CUDACC_FLAGS :=

# project files:
FILES := random_number_generator_kernal.cu  sample_run.cu

##################### Build Rules ########################random_number_generator_host.cu grid_block_test_case.cu performance_benchmark.cu

all: cuMain

clean: 
	rm -f cuMain

cuMain: main.cu
	$(CUDACC) $(CUDACC_FLAGS) -o $@ $< $(FILES)
	
##########################################################
