#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <CL/opencl.h>
#include <math.h>
#include <complex.h>
#include <time.h>

#define PI 3.14159265358979323846265358979323846

float complex** Make2DPtr(int size){
    float complex** ptr = (float complex**)calloc(size,sizeof(float complex*));
    for(int i = 0;i<size;i++){
        ptr[i] = (float complex*)calloc(size,sizeof(float complex));
    }
    return ptr;
}


float complex** StepCalc(float complex** psi_t,float complex** V,
    int res,float dx,float dy,
    float dt,float hbar,float m){
    float complex** psi_n = Make2DPtr(res);
    for(int i=0;i<res;i++){
        for(int j=0;j<res;j++){
            if(i==0 || i==res-1 || j==0 || j==res-1){
                psi_n[i][j]=0;
            }
            else{
                float complex s2psi1 = (psi_t[i+1][j]-2*psi_t[i][j]+psi_t[i-1][j])/(float complex)pow(dx,2.0);
                float complex s2psi2 = (psi_t[i][j+1]-2*psi_t[i][j]+psi_t[i][j-1])/(float complex)pow(dy,2.0);
                
                psi_n[i][j] = psi_t[i][j]-I*dt*(-(float complex)(hbar/(2*m))*(s2psi1+s2psi2)+V[i][j]*psi_t[i][j]);
            }
        }
    }
    return psi_n;
}