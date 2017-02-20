#include <linux/module.h>
#include <linux/types.h>
#include <linux/mm.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/device.h>
#include <linux/io.h>
#include <linux/cdev.h>    //cdev_init等函数定义
#include <asm/uaccess.h>
#include <linux/miscdevice.h>

#define DEVICE_NAME "misc_led"

volatile unsigned long *gpfcon = NULL;
volatile unsigned long *gpfdat = NULL;

static int led_ioctl(struct inode *inode, struct file *file, unsigned int cmd,unsigned long arg)
{
	if(cmd == 1)
	{
		//点亮
		*gpfdat &= ~((1<<4) | (1<<5) | (1<<6));
	}
	else
	{
		//熄灭
		*gpfdat |= (1<<4) | (1<<5) | (1<<6);
	}
}

static struct file_operations dev_fops = {
	.owner = THIS_MODULE,
	.ioctl = led_ioctl,
};

static struct miscdevice misc = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = DEVICE_NAME,
	.fops = &dev_fops,
};

static void __exit misc_dev_exit()
{
	iounmap(gpfcon);
	misc_deregister(&misc);
}

static int __init misc_dev_init(void)
{
	int ret;
	
	ret = misc_register(&misc);
	
	gpfcon = (volatile unsigned long *)ioremap(0x56000050, 16);
	gpfdat = gpfcon + 1;
	
	//配置GPF4,5,6为输出
	*gpfcon &= ~((0x3<<(4*2)) | (0x3<<(5*2)) | (0x3<<(6*2)));
	*gpfcon |= ((0x1<<(4*2)) | (0x1<<(5*2)) | (0x1<<(6*2)));
	
	return 0;
}

module_init(misc_dev_init);
module_exit(misc_dev_exit);

MODULE_LICENSE("GPL");