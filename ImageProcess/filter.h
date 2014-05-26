#include "CImg.h"

enum FILTER_TYPE{
	NONE,
	BLUR
};
float static filter_None[9] =  
{ 
     0, 0, 0, 
     0, 1, 0, 
     0, 0, 0 
};

float static filter_Blur[9] =
{
     0.0, 0.2,  0.0,
     0.2, 0.2,  0.2,
     0.0, 0.2,  0.0
};
float static filter_find_edgs[25] =
{
     0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,
    -1, -1,  2,  0,  0,
     0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,
};

float static filter_sharpen[9] =
{
    -1, -1, -1,
    -1,  9, -1,
    -1, -1, -1
};

