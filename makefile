all:
	rm -rf obj/*
	rm -rf results/*
	mkdir results/cuda
	mkdir results/cuda/diff
	mkdir results/cuda/matriz
	mkdir results/seq
	mkdir results/seq/diff
	mkdir results/seq/matriz
	nvcc src/diffusion.cu -o obj/diffusion -arch=sm_50
	gcc src/sequencial.c -o obj/sequencial
run:
	./tests/test_cuda.sh
	./tests/test_seq.sh
clean:
	rm -rf results/*
	mkdir results/cuda
	mkdir results/cuda/diff
	mkdir results/cuda/matriz
	mkdir results/seq
	mkdir results/seq/diff
	mkdir results/seq/matriz