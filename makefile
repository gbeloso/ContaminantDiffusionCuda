all:
	nvcc src/diffusion.cu -o obj/diffusion -arch=sm_50
	gcc src/sequencial.c -o obj/sequencial
run:
	./obj/diffusion 100 2000
	./obj/sequencial 100 2000