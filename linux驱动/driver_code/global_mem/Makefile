KERN_DIR = /home/jay/Code/JZ2440/linux-2.6.22.6
#EXTRA_CFLAGS = -g 
all:
	make -C $(KERN_DIR) M=`pwd` modules 
	arm-linux-gcc global_mem_drv_test.c -o global_mem_drv_test

clean:
	make -C $(KERN_DIR) M=`pwd` modules clean
	rm -rf modules.order

obj-m	+= global_mem_drv.o
