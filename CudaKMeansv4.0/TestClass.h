#ifndef TESTCLASS_H
#define TESTCLASS_H
#include <vector>


using namespace std;

class TestClass
{
public:
	TestClass(void);
	~TestClass(void);
	static void		TestData(float* fd, int nO, int nD);
	static void		TestMember(int* mem, int nO);
	static bool		CheckResult(int* CPU_member, int* CUDA_member
						  ,int nObj, int nCluster, int nDim);
	static float*	initData(int m, int n);
};

#endif