#include <linux/init.h>
#include <linux/module.h>
#include <linux/fs.h>
#include <asm/io.h>
#include <asm/uaccess.h>
#include <linux/device.h>
#include <linux/cdev.h>
#include <linux/platform_device.h>

#define DEVICE_NAME "led"

struct led_dev_t
{
	struct cdev cdev;
}led_dev;

static struct class	*led_drv_class;

int led_major;


volatile unsigned long *gpfcon = NULL; 
volatile unsigned long *gpfdat = NULL;

static int led_drv_ioctl(struct inode *inode, struct file *file, unsigned int cmd,unsigned long arg)
{
	if(cmd == 1)
	{
		//Ä³¸öµãÁÁ
		*gpfdat &= ~(0x1<<(arg + 4));
	}	
	else
	{
		//Ä³¸öÏ¨Ãð
		*gpfdat |= (0x1<<(arg + 4));
	}
}

static int led_drv_open()
{
	//ÅäÖÃGPF4,5,6ÎªÊä³ö
	*gpfcon &= ~((0x3<<(4*2)) | (0x3<<(5*2)) | (0x3<<(6*2)));
	*gpfcon |= ((0x1<<(4*2)) | (0x1<<(5*2)) | (0x1<<(6*2)));
	
	return 0;
}

static struct file_operations led_drv_fops = {
    .owner  =   THIS_MODULE,
    .open   =   led_drv_open,
	.ioctl  =   led_drv_ioctl,

};

static int __devinit led_probe(struct platform_device *pdev)
{
	printk("led_probe\n");
	
	dev_t dev_id;

	if(led_major)
	{
		//¾²Ì¬
		dev_id = MKDEV(led_major,0);
		register_chrdev_region(dev_id,1,DEVICE_NAME);
	}
	else
	{
		//¶¯Ì¬
		alloc_chrdev_region(&dev_id,0,1,DEVICE_NAME);
		led_major = MAJOR(dev_id);
	}
	
	cdev_init(&led_dev.cdev, &led_drv_fops);
	led_dev.cdev.owner = THIS_MODULE;
	cdev_add(&led_dev.cdev,dev_id,1);
	
	led_drv_class = class_create(THIS_MODULE, DEVICE_NAME);
	class_device_create(led_drv_class, NULL, MKDEV(led_major, 0), NULL, DEVICE_NAME);
	
	struct resource *pIORESOURCE_MEM = NULL;
	pIORESOURCE_MEM = platform_get_resource(pdev,IORESOURCE_MEM,0);
	
	gpfcon = ioremap(pIORESOURCE_MEM->start,pIORESOURCE_MEM->end - pIORESOURCE_MEM->start);
	gpfdat = gpfcon + 1;
		
	return 0;
}

static int __devexit led_remove(struct platform_device *pdev)
{
	printk("led_remove\n");

	class_device_destroy(led_drv_class,MKDEV(led_major,0));
	class_destroy(led_drv_class);
	cdev_del(&led_dev.cdev);
	unregister_chrdev_region(MKDEV(led_major,0),1);
	iounmap(gpfcon);
	
	return 0;
}

static struct platform_driver led_platform_driver = 
{
	.probe = led_probe,
	.remove = __devexit_p(led_remove),
	.driver = {
		.name = "jz2440 led",
		.owner = THIS_MODULE,
	}
};

static int __init led_platform_driver_init()
{
	printk("led_platform_driver_init\n");
	return platform_driver_register(&led_platform_driver);
}

static void __exit led_platform_driver_exit()
{
	printk("led_platform_driver_exit\n");
	platform_driver_unregister(&led_platform_driver);
}

module_init(led_platform_driver_init);
module_exit(led_platform_driver_exit);

MODULE_LICENSE("GPL");