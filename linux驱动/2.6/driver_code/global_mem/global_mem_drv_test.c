#include <stdio.h>
#include <signal.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

static void signalio_handler(int signum)
{	
	printf("receive a signal from globalmem,signal num:%d\n",signum);

}

int main()
{
	int fd,oflags;
	fd = open("/dev/global_mem", O_RDWR, S_IRUSR | S_IWUSR);
	if(fd != -1)
	{
		signal(SIGIO, signalio_handler);
		fcntl(fd, F_SETOWN, getpid());
		oflags = fcntl(fd, F_GETFL);
		fcntl(fd, F_SETFL, oflags | FASYNC);
		while(1)
		{
			sleep(100);
		}
	}
	else
	{
		printf("open fail\n");
	}	
}

