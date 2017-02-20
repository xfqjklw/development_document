#include <linux/module.h>
#include <linux/types.h>
#include <linux/mm.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/device.h>
#include <linux/io.h>
#include <linux/cdev.h>    //cdev_init等函数定义
#include <asm/uaccess.h>

#define USE_KMALLOC 0
#define MEMC_SIZE 4096
#define DEVICE_NAME "memc"

int memc_major;
char *memc_data;
static struct class	*memcdrv_class;

struct memc_dev_t
{
	struct cdev cdev;
}memc_dev;

static int memc_mmap(struct file*filp, struct vm_area_struct *vma)
{
	#if !USE_KMALLOC
	unsigned long pfn;
	int ret;
	unsigned long start = vma->vm_start;
	#endif
	
	unsigned long size = PAGE_ALIGN(vma->vm_end - vma->vm_start);
	printk("mmap size=%ld\n",size);	
	if(size > MEMC_SIZE) 
	{
		return -EINVAL;
	}

	vma->vm_flags |= VM_IO;
	vma->vm_flags |= VM_RESERVED;

	#if USE_KMALLOC
	if (remap_pfn_range(vma,vma->vm_start,virt_to_phys(memc_data)>>PAGE_SHIFT, size, vma->vm_page_prot))
		return  -EAGAIN;
	#else
	while (size > 0) 
	{
		pfn = vmalloc_to_pfn(memc_data);
		if ((ret = remap_pfn_range(vma, start, pfn, PAGE_SIZE, vma->vm_page_prot)) < 0) 
		{
			return ret;
		}
		start += PAGE_SIZE;
		memc_data += PAGE_SIZE;
		size -= PAGE_SIZE;
	}
	#endif
	
	return 0;
}

static int memc_open(struct inode *inode, struct file *filp)
{
	return 0;
}

static int memc_release(struct inode *inode, struct file *filp)
{
	return 0;
}

static const struct file_operations memc_fops =
{
  .owner = THIS_MODULE,
  .open = memc_open,
  .release = memc_release,
  .mmap = memc_mmap,
};

static int __init memc_init(void)
{
	dev_t dev_id;
	
	if(memc_major)
	{
		dev_id = MKDEV(memc_major,0);
		register_chrdev_region(dev_id,1,DEVICE_NAME);
	}	
	else
	{
		//动态
		alloc_chrdev_region(&dev_id,0,1,DEVICE_NAME);
		memc_major = MAJOR(dev_id);
	}
	
	cdev_init(&memc_dev.cdev, &memc_fops);
	memc_dev.cdev.owner = THIS_MODULE;
	cdev_add(&memc_dev.cdev,dev_id,1);
	
	memcdrv_class = class_create(THIS_MODULE, DEVICE_NAME);
	class_device_create(memcdrv_class, NULL, MKDEV(memc_major, 0), NULL, DEVICE_NAME);
	
	#if USE_KMALLOC
	memc_data = kzalloc(MEMC_SIZE, GFP_KERNEL);
	#else
	memc_data = vmalloc(MEMC_SIZE);
	memset(memc_data,0x00,MEMC_SIZE);
	#endif
	
	return 0;
}

static void __exit memc_exit()
{
	class_device_destroy(memcdrv_class,MKDEV(memc_major,0));
	class_destroy(memcdrv_class);
	cdev_del(&memc_dev.cdev);
	unregister_chrdev_region(MKDEV(memc_major,0),1);
	
	#if USE_KMALLOC
	kfree(memc_data);
	#else
	vfree(memc_data);
	#endif
}

module_init(memc_init);
module_exit(memc_exit);

MODULE_LICENSE("GPL");
