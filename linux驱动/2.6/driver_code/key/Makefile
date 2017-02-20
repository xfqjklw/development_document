KERN_DIR = /home/jay/Code/JZ2440/linux-2.6.22.6
#EXTRA_CFLAGS = -g 
all:
	make -C $(KERN_DIR) M=`pwd` modules 
	arm-linux-gcc key_drv_test.c -o key_drv_test

clean:
	make -C $(KERN_DIR) M=`pwd` modules clean
	rm -rf modules.order

obj-m	+= key_drv.o
