#ifndef NETPBM_H
#define NETPBM_H

#include <stdio.h>

typedef struct {
    FILE *fd;
    int   width;
    int   height;
    int   size;
    int   offset;
    char *data;
} netpbm;

typedef netpbm *netpbm_ptr;

int   open_image (char *path, netpbm_ptr img);
int   empty_image(char *path, netpbm_ptr img, int width, int height);
char *pixel_at   (netpbm_ptr img, int x, int y);
int   close_image(netpbm_ptr img);

#endif
