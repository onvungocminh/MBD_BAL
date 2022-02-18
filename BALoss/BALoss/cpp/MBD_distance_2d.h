#include <iostream>
#include <vector>
using namespace std;

struct Point2D
{
    // float distance;
    int w;
    int h;
};

void MBD_cut(const unsigned char * img, const unsigned char * seeds, const unsigned char * destination, unsigned char * distance, 
                              int height, int width);