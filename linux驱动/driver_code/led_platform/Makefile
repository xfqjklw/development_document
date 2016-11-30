KERN_DIR = /home/jay/Code/JZ2440/linux-2.6.22.6
#EXTRA_CFLAGS = -g 
all:
	make -C $(KERN_DIR) M=`pwd` modules 
	arm-linux-gcc led_platform_test.c -o led_platform_test

clean:
	make -C $(KERN_DIR) M=`pwd` modules clean
	rm -rf modules.order

obj-m	+= led_platform_device.o
obj-m   += led_platform_driver.o