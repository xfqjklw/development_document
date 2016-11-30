#include <linux/init.h>
#include <linux/module.h>
#include <linux/fs.h>
#include <asm/io.h>
#include <asm/uaccess.h>
#include <linux/device.h>
#include <linux/cdev.h>
#include <linux/platform_device.h>

static void led_platform_device_release(struct device *dev)
{
	return;
}

static struct resource led_resource[] =
{
	[0] =
	{
		.start = 0x56000050,
		.end   = 0x56000050+16,
		.flags = IORESOURCE_MEM,
	},
};

static struct platform_device led_platform_device =
{
	.name = "jz2440 led",
	.id   = -1,
	.num_resources = ARRAY_SIZE(led_resource),
	.resource = led_resource,
	.dev = {
		.release = led_platform_device_release,
	},
};

static int __init led_platform_device_init()
{
	printk("led_platform_device_init\n");
	return platform_device_register(&led_platform_device);
}

static void __exit led_platform_device_exit()
{	
	printk("led_platform_device_exit\n");
	platform_device_unregister(&led_platform_device);
}

module_init(led_platform_device_init);
module_exit(led_platform_device_exit);

MODULE_LICENSE("GPL");