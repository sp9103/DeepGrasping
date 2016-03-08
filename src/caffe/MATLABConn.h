#pragma once

#include <C:/Program Files (x86)/MATLAB/R2015b/extern/include/engine.h>
#include <C:/Program Files (x86)/MATLAB/R2015b/extern/include/mat.h>

#pragma comment(lib, "C:\\caffe-window-new\\caffe-windows-master (1)\\caffe-windows-master\\buildSpatial_add\\MSVC\\libmx.lib")
#pragma comment(lib, "C:\\caffe-window-new\\caffe-windows-master (1)\\caffe-windows-master\\buildSpatial_add\\MSVC\\libmat.lib")
#pragma comment(lib, "C:\\caffe-window-new\\caffe-windows-master (1)\\caffe-windows-master\\buildSpatial_add\\MSVC\\libeng.lib")

class MATLABConn
{
public:
	MATLABConn(void);
	~MATLABConn(void);

	bool initialize();
	double* ReadMatFile(const char *file, const char *name);

	//Engine *matlab;
	int m_nDatalength;
};

