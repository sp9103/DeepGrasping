#include "MATLABConn.h"

#define BUFSIZE 1000
MATLABConn::MATLABConn(void)
{

}


MATLABConn::~MATLABConn(void)
{
	//engClose(matlab);
}

bool MATLABConn::initialize()
{
	/*matlab = NULL;
	matlab = engOpen(NULL);
	
	if ( matlab == NULL)
		return false;*/

	return true;
}

double* MATLABConn::ReadMatFile(const char *file, const char *name)
{
	MATFile *pmat;
	mxArray *velData;
	int	  i, ndir, ndim;
	const char **dir;
	double* data = NULL;

	pmat = matOpen(file, "r");
	if (pmat == NULL) {
		printf("Error opening file %s\n", file);
		return data;
	}

//    pa = matGetNextVariableInfo(pmat, &name);
	velData = matGetVariable(pmat, name);
    if (velData == NULL) {
		printf("Error reading in file %s\n", file);
		return data;
	}

//	ndim = mxGetNumberOfDimensions(pa);

	mwSize nRow = mxGetM(velData);
	mwSize nCol = mxGetN(velData);

	m_nDatalength = nCol;

	double *pVal = (double*)mxGetPr(velData);

	data = new double[nCol*nRow];

	for (i = 0; i < nRow*nCol; i++)
		data[i] = *pVal++;

	mxDestroyArray(velData);

	if (matClose(pmat) != 0) {
		printf("Error closing file %s\n",file);
		return data;
	}

	return data;
}