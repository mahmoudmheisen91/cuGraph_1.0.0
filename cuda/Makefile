########## the compilers and their flags #################

# the compilers:
CUDACC := nvcc

# flags:
CUDACC_FLAGS :=

# project files:
FILES :=

##################### Build Rules ########################

all: cuMain

clean: 
	rm -f cuMain

cuMain: main.cu
	$(CUDACC) $(CUDACC_FLAGS) -o $@ $< $(FILES)
	
##########################################################
