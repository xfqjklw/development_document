#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>

int main(int argc, char **argv)
{
	char ch;
	int fd,fd0,fd1,fd2;
	int tmp;
	int tmpFd;
	
	int val = 0;
	char retChar[12];
	
	fd  = open("/dev/leds", O_RDWR);
	fd0 = open("/dev/led0", O_RDWR);
	fd1 = open("/dev/led1", O_RDWR);
	fd2 = open("/dev/led2", O_RDWR);

	printf("JZ2440 leds control\n");
	printf("1:all\n");
	printf("2:led0\n");
	printf("3:led1\n");
	printf("4:led2\n");
	while( (ch=getchar()) != '\n')
	{
		tmp = ch-48;
	}
		
	if(tmp == 1)
		tmpFd = fd;
	else if(tmp == 2)
		tmpFd = fd0;
	else if(tmp == 3)
		tmpFd = fd1;
	else if(tmp == 4)
		tmpFd = fd2;
	
	printf("1.on\n");
	printf("2.off\n");
	while( (ch=getchar()) != '\n')
	{
		tmp = ch-48;
	}
	
	if(tmp == 1)
		val = 1;
	else if(tmp == 2)
		val = 0;
	
	//write(tmpFd, &val, 4);
	ioctl(tmpFd, val ,0);
	
	int ret = read(tmpFd , retChar ,12);
	retChar[ret] = '\0';
	
	printf("ret=%d\n",ret);
	if(ret == 12)
	{
		printf("%d %d %d\n",*(int *)retChar,*(int *)(retChar+4),*(int *)(retChar+8));
	}
	else
	{
		printf("%d\n",*(int *)retChar);
	}
	
	
	return 0;
}
