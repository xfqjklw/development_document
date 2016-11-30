#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/device.h>
#include <linux/io.h>
#include <linux/cdev.h>   
#include <linux/irq.h>
#include <linux/delay.h>
#include <linux/workqueue.h>
#include <linux/interrupt.h>
#include <linux/poll.h>
#include <linux/errno.h>

#include <asm/uaccess.h>
#include <asm/arch/map.h> 
#include <asm/arch/regs-gpio.h>

#define DEVICE_NAME "global_mem"
#define GLOBAL_MEM_SIZE 4096
#define MEM_CLEAR 		1

struct _global_mem_dev
{
	struct cdev cdev;
	unsigned int current_len;
	unsigned char mem[GLOBAL_MEM_SIZE];
	struct mutex mutex;
	wait_queue_head_t r_wait;
	wait_queue_head_t w_wait;
	struct fasync_struct *async_queue;
}global_mem_dev;

static struct class *global_mem_drv_class;


static int global_mem_drv_fasync(int fd, struct file* file, int mode)
{
	struct _global_mem_dev *dev = file->private_data;
	return fasync_helper(fd,file,mode,&dev->async_queue);
}

static int global_mem_drv_open(struct inode *inode, struct file *file)
{
	struct _global_mem_dev *devp;
	devp = container_of(inode->i_cdev,struct _global_mem_dev,cdev);
	file->private_data = devp;
	
	return 0;
}

static int global_mem_drv_close(struct inode *inode, struct file *file)
{
	global_mem_drv_fasync(-1, file, 0);
	return 0;
}

static ssize_t global_mem_drv_write(struct file *file, const char __user *buf, size_t count, loff_t * ppos)
{
	int ret;
	struct _global_mem_dev *dev = file->private_data;
	DECLARE_WAITQUEUE(wait,current); //声明等待队列
	
	mutex_lock(&dev->mutex);
	add_wait_queue(&dev->r_wait,&wait);	//加入读等待队列头
	
	//如果队列已经满了
	while(dev->current_len == GLOBAL_MEM_SIZE)
	{
		if(file->f_flags & O_NONBLOCK)
		{
			ret = -EAGAIN;
			goto out;
		}
		__set_current_state(TASK_INTERRUPTIBLE);
		mutex_unlock(&dev->mutex);
		
		schedule();
		if(signal_pending(current))
		{
			ret = -ERESTARTSYS;
			goto out2;
		}
		
		//如果mem满了，在睡眠状态被被读操作唤醒
		mutex_lock(&dev->mutex);
	}
	
	//如果写入的数据超过了mem的最大值，则只写入部分
	if(count > GLOBAL_MEM_SIZE - dev->current_len)
		count = GLOBAL_MEM_SIZE - dev->current_len;
	
	if(copy_from_user(dev->mem+dev->current_len, buf, count))
	{
		ret = -EFAULT;
		goto out;
	}
	else
	{
		//如果写入成功，更新长度
		dev->current_len += count;
		printk("write %d bytes,current_len %d\n",count,dev->current_len);
		
		//如果读操作因为mem为空阻塞，则会被该写操作唤醒
		wake_up_interruptible(&dev->r_wait);
		
		if(dev->async_queue)
		{
			kill_fasync(&dev->async_queue, SIGIO, POLLIN);
			printk(KERN_DEBUG "%s kill SIGIO",__func__);
		}
		
		ret = count;
	}
	
	out:
	mutex_unlock(&dev->mutex);
	out2:
	remove_wait_queue(&dev->w_wait,&wait);
	set_current_state(TASK_RUNNING);
	
	return ret;
}

static ssize_t global_mem_drv_read(struct file *file, char *buf, size_t len, loff_t *off)
{
	int ret;
	struct _global_mem_dev *dev = file->private_data;
	DECLARE_WAITQUEUE(wait,current); //声明等待队列
	
	mutex_lock(&dev->mutex);
	add_wait_queue(&dev->r_wait,&wait);	//加入读等待队列头
	
	while(dev->current_len == 0)
	{
		if(file->f_flags & O_NONBLOCK)
		{
			ret = -EAGAIN;
			goto out;
		}
		//如果mem中没有数据，设置进程状态为休眠
		__set_current_state(TASK_INTERRUPTIBLE);
		mutex_unlock(&dev->mutex);
		
		schedule();//进程调度，休眠
		//如果被信号唤醒，则退出
		if(signal_pending(current))
		{
			ret = -ERESTARTSYS;
			goto out2;
		}
		//否则被写操作唤醒
		//还要继续判断current是否为0，上锁
		mutex_lock(&dev->mutex);
		
	}	
	
	if(len > dev->current_len)
		len = dev->current_len;
	
	if(copy_to_user(buf,dev->mem,len))
	{
		//如果拷贝给用户失败，退出
		ret = -EFAULT;
		goto out;
	}
	else
	{
		//如果拷贝给用户成功，重新移动mem中的数据
		memcpy(dev->mem,dev->mem+len,dev->current_len-len);
		dev->current_len -= len;
		printk("read %d bytes,current len:%d\n",len,dev->current_len);
		
		//如果读出了一些数据，写操作处于阻塞时，可以唤醒写操作
		wake_up_interruptible(&dev->w_wait);
		ret = len;
	}
	
	out:
	mutex_unlock(&dev->mutex);
	out2:
	remove_wait_queue(&dev->r_wait,&wait);
	set_current_state(TASK_RUNNING);
	
	return ret;

}

static int global_mem_drv_ioctl(struct inode *inode, struct file *file, unsigned int cmd,unsigned long arg)
{
	struct _global_mem_dev *dev = file->private_data;
	switch(cmd)
	{
		case MEM_CLEAR:
			mutex_lock(&dev->mutex);
			memset(dev->mem,0,GLOBAL_MEM_SIZE);
			mutex_unlock(&dev->mutex);
			printk("global_mem clear\n");
			break;
	}
	
	return 0;
}

static int global_mem_drv_poll(struct file *file, struct poll_table_struct *wait)
{
	unsigned int mask = 0;
	struct _global_mem_dev *dev = file->private_data;
	
	mutex_lock(&dev->mutex);
	
	poll_wait(file, &dev->r_wait, wait);
	poll_wait(file, &dev->w_wait, wait);
	
	if(dev->current_len != 0)
	{
		mask |= POLLIN | POLLRDNORM;
	}
	
	if(dev->current_len != GLOBAL_MEM_SIZE)
	{
		mask |= POLLOUT | POLLWRNORM;
	}
	
	mutex_unlock(&dev->mutex);
	
	return mask;
}

static struct file_operations global_mem_drv_fops = {
    .owner   =  THIS_MODULE,
    .open    =  global_mem_drv_open,
	.write	 =  global_mem_drv_write,
	.read	 =	global_mem_drv_read,
	.release =  global_mem_drv_close, 
	.poll    =  global_mem_drv_poll,
	.ioctl	 =  global_mem_drv_ioctl,
	.fasync  =  global_mem_drv_fasync,
};

int global_mem_major;

static int global_mem_drv_init(void)
{
	dev_t dev_id;

	//1.申请设备号
	//注册字符设备编号，在proc/devices中创建global_mem选项
	if(global_mem_major)
	{
		//静态
		dev_id = MKDEV(global_mem_major,0);
		register_chrdev_region(dev_id,1,DEVICE_NAME);
	}
	else
	{
		//动态
		alloc_chrdev_region(&dev_id,0,1,DEVICE_NAME);
		global_mem_major = MAJOR(dev_id);
	}

	//2.注册字符设备
	//初始化并且注册keys cdev
	cdev_init(&global_mem_dev.cdev, &global_mem_drv_fops);
	global_mem_dev.cdev.owner = THIS_MODULE;
	cdev_add(&global_mem_dev.cdev,dev_id,1);
	
	//3.自动创建设备节点
	//创建一个类，这个类存放于sysfs下面 /sys/class/keys
	global_mem_drv_class = class_create(THIS_MODULE, DEVICE_NAME);
	
	//mdev daemon就会自动创建节点/dev/keys
	//加载模块的时候，用户空间中的mdev会自动响应device_create(…)函数，去/sysfs下寻找对应的类从而创建设备节点。
	class_device_create(global_mem_drv_class, NULL, MKDEV(global_mem_major, 0), NULL, DEVICE_NAME);
	
	//初始化等待队列
	init_waitqueue_head(&global_mem_dev.r_wait);
	init_waitqueue_head(&global_mem_dev.w_wait);

	//初始化互斥锁
	mutex_init(&global_mem_dev.mutex);
	
	global_mem_dev.current_len = 0;
	
	return 0;
}


static void global_mem_drv_exit(void)
{
	//1.删除设备节点
	//删除/dev/keys设备节点
	class_device_destroy(global_mem_drv_class,MKDEV(global_mem_major,0));
	class_destroy(global_mem_drv_class);

	//2.删除字符设备
	//删除cdev
	cdev_del(&global_mem_dev.cdev);
	
	//3.释放设备号
	unregister_chrdev_region(MKDEV(global_mem_major,0),1);
}

module_init(global_mem_drv_init);
module_exit(global_mem_drv_exit);
MODULE_LICENSE("GPL");
