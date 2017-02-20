#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/device.h>
#include <linux/io.h>
#include <linux/cdev.h>    //cdev_init等函数定义
#include <asm/uaccess.h>
#include <asm/arch/map.h>  //包含了S3C24XX_VA_GPIO定义

#define DEVICE_NAME "leds"
#define MAX_LEDS_NUM 3

//leds结构体
//status: 1 亮 0 灭
struct leds_dev_t
{
	struct cdev cdev;
	int status[MAX_LEDS_NUM];
}leds_dev;

//led1/2/3结构体
//status: 1 亮 0 灭
struct led_dev_t
{
	struct cdev cdev;
	int status;
} *led_dev;

//class结构体，用于自动生成设备节点
static struct class 		*ledsdrv_class;

//gpio寄存器虚拟地址
volatile unsigned long *gpfcon = NULL;
volatile unsigned long *gpfdat = NULL;

//驱动主设备号
static int leds_major;

static int leds_drv_ioctl(struct inode *inode, struct file *file, unsigned int cmd,unsigned long arg)
{
	
	int i;
	int minor = iminor(inode);
	printk("leds_drv_ioctl,minor:%d\n",minor);
	
	if(minor == 0)
	{
		struct leds_dev_t *dev = file->private_data;
		if (cmd == 1)
		{
			//全部点亮
			*gpfdat &= ~((1<<4) | (1<<5) | (1<<6));
			for(i = 0; i <  MAX_LEDS_NUM; i++)
			{
				dev->status[i] = 1;
			}	
		}
		else
		{
			//全部熄灭
			*gpfdat |= (1<<4) | (1<<5) | (1<<6);
			for(i = 0; i <  MAX_LEDS_NUM; i++)
			{
				dev->status[i] = 0;
			}
		}
	}	
	else
	{	
		struct led_dev_t *dev = file->private_data;
		if(cmd == 1)
		{
			//某个点亮
			*gpfdat &= ~(0x1<<(minor + 3));
			dev->status = 1;
			leds_dev.status[minor-1] = 1;
		}
		else
		{
			//某个熄灭
			*gpfdat |= (0x1<<(minor + 3));
			dev->status = 0;
			leds_dev.status[minor-1] = 0;
		}
	}
}
		


static int leds_drv_open(struct inode *inode, struct file *file)
{
	struct leds_dev_t *dev1;
	struct led_dev_t  *dev2;

	//获取从设备号
	int minor = iminor(inode);
	printk("leds_drv_open,minor:%d\n",minor);
	
	if(minor == 0)
	{
		//用struct file的文件私有数据指针保存struct mycdev结构体指针
		dev1 = container_of(inode->i_cdev,struct leds_dev_t,cdev);
		file->private_data = dev1;
		
		//配置GPF4,5,6为输出
		*gpfcon &= ~((0x3<<(4*2)) | (0x3<<(5*2)) | (0x3<<(6*2)));
		*gpfcon |= ((0x1<<(4*2)) | (0x1<<(5*2)) | (0x1<<(6*2)));
	}	
	else
	{	
		//用struct file的文件私有数据指针保存struct mycdev结构体指针
		dev2=container_of(inode->i_cdev,struct led_dev_t,cdev);
		file->private_data = dev2;
		
		//否则单独设置某个引脚为输出功能
		*gpfcon &= ~(0x3 << ((minor+3)*2)); 
		*gpfcon |= (0x1 << ((minor+3)*2));
	}	

	return 0;
}

static ssize_t leds_drv_write(struct file *file, const char __user *buf, size_t count, loff_t * ppos)
{
	int i;
	int val;

	int minor = iminor(file->f_path.dentry->d_inode);
	printk("leds_drv_write,minor:%d\n",minor);
	
	copy_from_user(&val, buf, count);

	if(minor == 0)
	{
		//获取私有数据
		struct leds_dev_t *dev = file->private_data;
		if (val == 1)
		{
			//全部点亮
			*gpfdat &= ~((1<<4) | (1<<5) | (1<<6));
			for(i = 0; i <  MAX_LEDS_NUM; i++)
			{
				dev->status[i] = 1;
			}	
		}
		else
		{
			//全部熄灭
			*gpfdat |= (1<<4) | (1<<5) | (1<<6);
			for(i = 0; i <  MAX_LEDS_NUM; i++)
			{
				dev->status[i] = 0;
			}
		}
	}
	else
	{
		struct led_dev_t *dev = file->private_data;
		if(val == 1)
		{
			//某个点亮
			*gpfdat &= ~(0x1<<(minor + 3));
			dev->status = 1;
			leds_dev.status[minor-1] = 1;
		}
		else
		{
			//某个熄灭
			*gpfdat |= (0x1<<(minor + 3));
			dev->status = 0;
			leds_dev.status[minor-1] = 0;
		}
	}
			
	return 0;
}

ssize_t leds_drv_read(struct file *file, char *buf, size_t len, loff_t *off)
{
	int minor = iminor(file->f_path.dentry->d_inode);
	printk("leds_drv_read,minor:%d\n",minor);

	if(minor == 0)
	{
		struct leds_dev_t *dev = file->private_data;
		copy_to_user(buf, dev->status, MAX_LEDS_NUM*sizeof(int));
		return (MAX_LEDS_NUM*sizeof(int));
	}
	else
	{
		struct led_dev_t *dev = file->private_data;
		copy_to_user(buf, &dev->status, sizeof(int));
		return sizeof(int);
	}
	
}

//led字符设备驱动对应的操作函数
static struct file_operations leds_drv_fops = {
    .owner  =   THIS_MODULE,
    .open   =   leds_drv_open,
	.write	=	leds_drv_write,
	.read	=	leds_drv_read,
	.ioctl  =   leds_drv_ioctl,
};


static int leds_drv_init(void)
{	
	int i;
	dev_t dev_id;
	
	//1.申请设备号
	//注册字符设备编号，在proc/devvices中创建leds选项
	if(leds_major)
	{
		//静态
		dev_id = MKDEV(leds_major,0);
		register_chrdev_region(dev_id,(1+MAX_LEDS_NUM),DEVICE_NAME);
	}
	else
	{
		//动态
		alloc_chrdev_region(&dev_id,0,(1+MAX_LEDS_NUM),DEVICE_NAME);
		leds_major = MAJOR(dev_id);
	}
	
	//2.注册字符设备
	//初始化并且注册leds cdev
	cdev_init(&leds_dev.cdev, &leds_drv_fops);
	leds_dev.cdev.owner = THIS_MODULE;
	cdev_add(&leds_dev.cdev,dev_id,1);
	
	//初始化并注册led0/1/2/3
	led_dev = kmalloc(MAX_LEDS_NUM*sizeof(struct led_dev_t),GFP_KERNEL);
	for(i = 0; i < MAX_LEDS_NUM; i++)
	{
		dev_id = MKDEV(leds_major,i+1);
		cdev_init(&led_dev[i].cdev, &leds_drv_fops);
		cdev_add(&led_dev[i].cdev, dev_id,1);
	}
	
	//3.自动创建设备节点
	//创建一个类，这个类存放于sysfs下面 /sys/class/leds
	ledsdrv_class = class_create(THIS_MODULE, DEVICE_NAME);
	
	//mdev daemon就会自动创建节点/dev/leds
	//加载模块的时候，用户空间中的mdev会自动响应device_create(…)函数，去/sysfs下寻找对应的类从而创建设备节点。
	class_device_create(ledsdrv_class, NULL, MKDEV(leds_major, 0), NULL, DEVICE_NAME);
	
	//创建节点/dev/led1/2/3
	for(i = 0; i < MAX_LEDS_NUM; i++)
	{
		class_device_create(ledsdrv_class, NULL, MKDEV(leds_major, i+1), NULL, "led%d", i);
	}
	
	//4.虚拟地址动态映射
	gpfcon = (volatile unsigned long *)ioremap(0x56000050, 16);
	gpfdat = gpfcon + 1;
	
	//也可以采用静态的方法
	//gpfcon = S3C24XX_VA_GPIO+0x50;
	//gpfdat = gpfcon + 1;

	return 0;
}

static void leds_drv_exit(void)
{
	int i;

	//1.删除设备节点
	//删除/dev/leds设备节点
	class_device_destroy(ledsdrv_class,MKDEV(leds_major,0));
	
	//删除/dev/led1/2/3设备节点
	for(i = 0; i < MAX_LEDS_NUM; i++)
	{
		class_device_destroy(ledsdrv_class,MKDEV(leds_major,i+1));
	}
	class_destroy(ledsdrv_class);

	//2.删除字符设备
	//删除cdev
	cdev_del(&leds_dev.cdev);
	for(i = 0; i < MAX_LEDS_NUM; i++)
	{
		cdev_del(&led_dev[i].cdev);
	}
	
	//3.释放设备号
	unregister_chrdev_region(MKDEV(leds_major,0),1+MAX_LEDS_NUM);
	
	//4.释放虚拟io空间
	iounmap(gpfcon);
	
	kfree(led_dev);
}

module_init(leds_drv_init);
module_exit(leds_drv_exit);

MODULE_LICENSE("GPL");

