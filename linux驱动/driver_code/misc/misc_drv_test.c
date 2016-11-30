#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>

void main()
{
	int fd;
	fd = open("/dev/misc_led",O_RDWR);
	
	ioctl(fd, 1 ,0);
	

}