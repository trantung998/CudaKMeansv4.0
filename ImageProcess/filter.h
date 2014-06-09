#include "CImg.h"
typedef struct 
{
	UINT filter_size;
	float* filter_matrix;
	float bias;
	float factor;
}filter_struct;
float static filter_None[9] =  
{ 
     1,  1,  1,
     1, -7,  1,
     1,  1,  1
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

float static filter_motion[81] =
{
    1, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1,
};

float static filter_sharpen_2[25] =
{
    -1, -1, -1, -1, -1,
    -1,  2,  2,  2, -1,
    -1,  2,  8,  2, -1,
    -1,  2,  2,  2, -1,
    -1, -1, -1, -1, -1,
};

class filter{
	filter_struct m_filter;
public: 
	filter();
	~filter();
	static filter_struct get_filter(){
		UINT r = rand()%5;
		printf("rand : %d\n",r);
		filter_struct fs;
		switch (r)
		{
		case 0:
		{
			fs.filter_size = 3;
			fs.bias = 0;
			fs.factor = 1;
			fs.filter_matrix = (float*)malloc(sizeof(float)*fs.filter_size*fs.filter_size);
			for(int i = 0 ; i <fs.filter_size*fs.filter_size;i++ )
			{
				fs.filter_matrix[i]= filter_Blur[i];
			}
		}
		break;
		case 1:
		{
			fs.filter_size = 5;
			fs.bias = 0;
			fs.factor = 1;
			fs.filter_matrix = (float*)malloc(sizeof(float)*fs.filter_size*fs.filter_size);
			for(int i = 0 ; i <fs.filter_size*fs.filter_size;i++ )
			{
				fs.filter_matrix[i]= filter_find_edgs[i];
			}
		}
		break;
		case 2:
		{
			fs.filter_size = 3;
			fs.bias = 0;
			fs.factor = 1;
			fs.filter_matrix = (float*)malloc(sizeof(float)*fs.filter_size*fs.filter_size);
			for(int i = 0 ; i <fs.filter_size*fs.filter_size;i++ )
			{
				fs.filter_matrix[i]= filter_sharpen[i];
			}
		}
		break;
		case 3:
		{
			fs.filter_size = 9;
			fs.bias = 0;
			fs.factor = 0.111111111111;
			fs.filter_matrix = (float*)malloc(sizeof(float)*fs.filter_size*fs.filter_size);
			for(int i = 0 ; i <fs.filter_size*fs.filter_size;i++ )
			{
				fs.filter_matrix[i]= filter_motion[i];
			}
		}
		break;
		case 4:
		{
			fs.filter_size = 5;
			fs.bias = 0;
			fs.factor = 0.125;
			fs.filter_matrix = (float*)malloc(sizeof(float)*fs.filter_size*fs.filter_size);
			for(int i = 0 ; i <fs.filter_size*fs.filter_size;i++ )
			{
				fs.filter_matrix[i]= filter_sharpen_2[i];
			}
		}
		break;
		default:
		break;
		}
		return fs;	
	}
};
enum FILTER_TYPE{
	NONE,
	BLUR
};


