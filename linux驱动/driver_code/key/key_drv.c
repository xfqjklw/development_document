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

#include <asm/uaccess.h>
#include <asm/arch/map.h> 
#include <asm/arch/regs-gpio.h>

#define DEVICE_NAME "keys"
#define MAX_KEY_NUM 	4
#define KEY_DOWN		1//按键按下 
#define KEY_UP 			2//按键抬起 
#define KEY_UNCERTAIN 	3//按键不确定

struct keys_dev_t
{
	struct cdev cdev;

}keys_dev;

static volatile int key_value;						//确认好的键值
static volatile int key_status[MAX_KEY_NUM];		//记录按键状态
static volatile int key_count[MAX_KEY_NUM];			//记录长按计数
static volatile int key_last[MAX_KEY_NUM];			//双击判断
static struct timer_list keys_timer[MAX_KEY_NUM];	//按键去抖定时器
static struct timer_list keys_double_timer[MAX_KEY_NUM];	//双击判断定时器

static struct class *keysdrv_class;

static DECLARE_WAIT_QUEUE_HEAD(button_waitq);

/* 中断事件标志, 中断服务程序将它置1，key_drv_read将它清0 */
static volatile int ev_press = 0;

struct pin_desc{
	unsigned int pin;
	unsigned int irq;
	char name[8];
};

struct pin_desc pins_desc[4] = {
	{S3C2410_GPF0, IRQ_EINT0, "key1"},
	{S3C2410_GPF2, IRQ_EINT2, "key2"},
	{S3C2410_GPG3, IRQ_EINT11,"key3"},
	{S3C2410_GPG11,IRQ_EINT19,"key4"},
};

struct work_struct my_wq;/*定义一个工作队列*/
void my_wq_func(unsigned long);/*定义一个处理函数*/

//中断处理底半部
void my_wq_func(unsigned long t)
{
	printk("one key preesed:%d\n",key_value);
}

static void keys_double_timer_handle(unsigned long data)
{
	//等待500ms后，如果定时器还未被删除，则认为是短按
	int key = data;
	
	key_last[key] = 0;
	key_status[key] = KEY_UP;
	key_count[key] = 0;
	ev_press = 1;
	schedule_work(&my_wq);//启动后半段
	wake_up_interruptible(&button_waitq);  /* 唤醒休眠的进程 */
}

static void keys_timer_handle(unsigned long data)
{
	
	int key = data;
	int ret;
	unsigned int pinval;
	
	pinval = s3c2410_gpio_getpin(pins_desc[key].pin);

	if(pinval)
	{
		/* 松开 */
		if(key_status[key] == KEY_UNCERTAIN)
		{
			//如果松开了，检测count的值，如果大于50(5秒)，
			//则认为是长按
			if(key_count[key] > 50)
			{
				key_value = key+10;			//长按的键值
				key_status[key] = KEY_UP;
				key_count[key] = 0;
				ev_press = 1;
				schedule_work(&my_wq);//启动后半段
				wake_up_interruptible(&button_waitq);  /* 唤醒休眠的进程 */
				
			}
			else 
			{
				//如果松开了，count<50,则认为是短按或者双击
				if(key_last[key] == 1)
				{
					//如果500ms内又一次被按下
					del_timer(&keys_double_timer[key]);
					key_last[key] = 0;
					key_value = key+20;
					
					key_status[key] = KEY_UP;
					key_count[key] = 0;
					ev_press = 1;
					schedule_work(&my_wq);//启动后半段
					wake_up_interruptible(&button_waitq);  /* 唤醒休眠的进程 */
				}
				else
				{
					//短按，但是并不能确定接下来的500ms会不会再次按下，形成双击
					//所以等待500ms，看看是否会按下。
					key_value = key;
					key_last[key] = 1;
					mod_timer(&keys_double_timer[key], jiffies+HZ/2);
				}
			}	
			
		}
 	}
	else
	{
		/* 按下 */
		if(key_status[key] == KEY_UNCERTAIN)
		{
			//确认按键已经按下了
			key_status[key] = KEY_DOWN;
			//继续启动定时器，判断是否长按
			mod_timer(&keys_timer[key], jiffies+HZ/100);
		}
		else if(key_status[key] == KEY_DOWN)
		{
			//如果定时器到了，还处于按下的状态，count++
			//并继续重启定时器
			mod_timer(&keys_timer[key], jiffies+HZ/100);
			//printk("%d\n",key_count[key]);
			key_count[key]++;
		}	
	}
}

/*
  * 确定按键值
  */
static irqreturn_t buttons_irq(int irq, void *dev_id)
{
	int key = (int)dev_id;
		
	//设置当前按键的状态为不确定
	key_status[key] = KEY_UNCERTAIN;
	//设置当前按键按下去抖定时器的延时并启动定时器
	mod_timer(&keys_timer[key], jiffies+HZ/100); //设置100ms超时 1HZ=1S
			
	return IRQ_RETVAL(IRQ_HANDLED);
}


ssize_t keys_drv_read(struct file *file, char __user *buf, size_t size, loff_t *ppos)
{
	if (size != 1)
		return -EINVAL;
	
	//判断非阻塞打开
	if(file->f_flags & O_NONBLOCK)
	{
		if(ev_press == 0)
			return -EAGAIN;
	}
	else
	{
		/* 如果没有按键动作, 休眠 */
		//ev_press为0时，休眠
		wait_event_interruptible(button_waitq, ev_press);
	}
	
	/* 如果有按键动作, 返回键值 */
	copy_to_user(buf, &key_value, sizeof(key_value));
	ev_press = 0;
	
	return 0;
}

static int keys_drv_open(struct inode *inode, struct file *file)
{
	/* 配置GPF0,2为输入引脚 */
	/* 配置GPG3,11为输入引脚 */
	//申请中断
	int i;
	for(i = 0; i < MAX_KEY_NUM; i++)
	{
		request_irq(pins_desc[i].irq,  buttons_irq, IRQT_BOTHEDGE, pins_desc[i].name, (void *)i);
		key_status[i] = KEY_UP;
		key_count[i] = 0;
		key_last[i] = 0;
		setup_timer(&keys_timer[i], keys_timer_handle, i);
		setup_timer(&keys_double_timer[i], keys_double_timer_handle, i);
	}

	return 0;
}

int keys_drv_close(struct inode *inode, struct file *file)
{
	int i;
	for(i = 0; i < MAX_KEY_NUM; i++)
	{
		free_irq(pins_desc[i].irq,  (void *)i);
		del_timer(&keys_timer[i]);
		del_timer(&keys_double_timer[i]);
	}	

	return 0;
}

static int keys_drv_poll(struct file *file, struct poll_table_struct *wait)
{
	unsigned int mask = 0;

	//添加等待队列到等待队列表中(poll_table)
	poll_wait(file, &button_waitq, wait);
	
	if(ev_press)
	{
		//标识数据可以获得
		mask |= POLLIN | POLLRDNORM;
	}

	return mask;
} 

static struct file_operations keys_drv_fops = {
    .owner   =  THIS_MODULE,    /* 这是一个宏，推向编译模块时自动创建的__this_module变量 */
    .open    =  keys_drv_open,
	.read	 =	keys_drv_read,
	.release =  keys_drv_close,	 
	.poll    =  keys_drv_poll,
};

int keys_major;
static int keys_drv_init(void)
{
	dev_t dev_id;

	//1.申请设备号
	//注册字符设备编号，在proc/devices中创建keys选项
	if(keys_major)
	{
		//静态
		dev_id = MKDEV(keys_major,0);
		register_chrdev_region(dev_id,1,DEVICE_NAME);
	}
	else
	{
		//动态
		alloc_chrdev_region(&dev_id,0,1,DEVICE_NAME);
		keys_major = MAJOR(dev_id);
	}

	//2.注册字符设备
	//初始化并且注册keys cdev
	cdev_init(&keys_dev.cdev, &keys_drv_fops);
	keys_dev.cdev.owner = THIS_MODULE;
	cdev_add(&keys_dev.cdev,dev_id,1);
	
	//3.自动创建设备节点
	//创建一个类，这个类存放于sysfs下面 /sys/class/keys
	keysdrv_class = class_create(THIS_MODULE, DEVICE_NAME);
	
	//mdev daemon就会自动创建节点/dev/keys
	//加载模块的时候，用户空间中的mdev会自动响应device_create(…)函数，去/sysfs下寻找对应的类从而创建设备节点。
	class_device_create(keysdrv_class, NULL, MKDEV(keys_major, 0), NULL, DEVICE_NAME);
	
	INIT_WORK(&my_wq,(void (*)(void*))my_wq_func);/*初始化工作队列并将其与处理函数绑定*/
	
	return 0;
}

static void keys_drv_exit(void)
{
	//1.删除设备节点
	//删除/dev/keys设备节点
	class_device_destroy(keysdrv_class,MKDEV(keys_major,0));
	class_destroy(keysdrv_class);

	//2.删除字符设备
	//删除cdev
	cdev_del(&keys_dev.cdev);
	
	//3.释放设备号
	unregister_chrdev_region(MKDEV(keys_major,0),1);
		
	return;
}

module_init(keys_drv_init);
module_exit(keys_drv_exit);
MODULE_LICENSE("GPL");
