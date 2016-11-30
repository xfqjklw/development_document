#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>

int main(int argc, char **argv)
{
	int fd;
	unsigned int key_val;
	
	fd = open("/dev/keys", O_RDWR);
	if (fd < 0)
	{
		printf("can't open!\n");
	}
	
	while (1)
	{
		int i;
		int ret;
		fd_set rds;
		
		FD_ZERO(&rds);
		FD_SET(fd, &rds);
		
		ret = select(fd + 1, &rds, NULL, NULL, NULL);
		
		if(ret < 0)
		{
			printf("Read Buttons Device Faild!\n");
			exit(1);
		}
		if(ret == 0)
		{
			printf("Read Buttons Device Timeout!\n");
		}
		else if(FD_ISSET(fd, &rds))
		{
			read(fd, &key_val, 1);
			printf("key_val = %d\n", key_val);
			}
	}
	
	close(fd);
	return 0;
}
