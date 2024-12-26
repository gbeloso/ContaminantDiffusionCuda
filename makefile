all:
	nvcc src/diffusion.cu -o obj/diffusion -arch=sm_50
	gcc src/sequencial.c -o obj/sequencial
run:
	./obj/diffusion 2000 100
	./obj/sequencial 2000 100