
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <cmath> 
#include<fstream>

/* ### NOTA BENE #####
# - Nel file cpp ci vuole extern "C"
# - Compilato a due passi:
#       g++ -fPIC -o3 -fopenmp -c matrix.cpp -o matrix.o
#       g++ -shared -o matrix.so matrix.o -fopenmp
*/

extern "C"{

/* Funzioni ausiliarie: */
/* Estrazione tempo massimo dal vettore globale dei tempi */
int find_t_min(float* T, float t1, int dim)
{
    int count = 0;
    
    while (T[count] - t1 <= 0)
    {
        count += 1;
    }
    if (std::abs(T[count]-t1) < 0.5)
        {count-=1;}
    return count;
}

/* Calcolo valore massimo di un vettore */
float max(float *v, int size)
{
    /*if (size <= 0)
        {throw("La taglia del vettore deve essere positiva");}*/
    float max = v[0];
    for (int i = 1; i < size; i++)
    {
        if (v[i] > max)
        {max = v[i];}
    }
    return max;
}

/* Calcolo valore minimo di un vettore */
float min(float *v, int size)
{
    /*if (size <= 0)
        {throw("La taglia del vettore deve essere positiva");}*/
    float min = v[0];
    for (int i = 1; i < size; i++)
    {
        if (v[i] < min)
        {min = v[i];}
    }
    return min;
}

/* Funzione per il calcolo della sinc traslata e pesata */
void sinc(float* t, int dim, float res, float* s, float dt, float d, int row, float point)
{
    /*if (dim <= 0 || row <= 0)
        {throw("La taglia del vettore deve essere positiva");}*/
    
    float soft = (point/sqrt(pow(point,2)+pow(d,2)));

    for (int i = 0; i < dim; i++)
    {  
        float aux = t[i] - (res/dt);
        if (aux == 0)
            {s[i] = 1/soft/d;}
        else
            {
                float arg = M_PI*aux;
                s[i] = sin(arg)/(arg)*soft/d;
            }
    }
}

void hmat(float* centri, int row, float* point, int np, float vel, float dt, float* geom, int NL, float* im, float* times){
    
    #pragma omp parallel for
    for (int m = 0; m < np; m++){
        
    /* Calcoli preliminari */
    float* d = (float*) calloc(row, sizeof(float));
    float* tpixel = (float*) calloc(row, sizeof(float));
    float* rit = (float*) calloc(row, sizeof(float));
    float aux1, aux2;
    #pragma omp parallel for
    for (int i = 0; i < row; i++)
    {
        aux1 = sqrt(pow(centri[i]-point[0+m*3],2)+pow(centri[i+row]-point[1+m*3],2)+pow(centri[i+2*row]-point[2+m*3],2));
        aux2 = aux1/vel;
        d[i] = aux1;
        tpixel[i] = aux2;
        rit[i] = aux2-geom[i];
    }
    
    float trif = min(rit, row);
    float tmax = max(rit, row);
    
    int dim = 2*NL+1;
    
    float tsin[dim];
    
    for(int k = 0; k<dim;k++)
        {tsin[k] = -NL+k;}
        
    float t1 = dt*round((trif-NL*dt)/dt);
    float t2 = dt*round((tmax+(NL+1)*dt)/dt);
    int ntempi = round((t2-t1)/dt)+1;

    float* t = (float*) calloc(ntempi, sizeof(float));
    for (int k = 0; k < ntempi; k++)
    {
        t[k] = t1+k*dt;
    }
    
    times[m] = t[0];
    float s[dim];

    for (int j = 0; j < row; j++)
    {
        
        float tgrid = dt*round(rit[j]/dt);
        float res = tgrid-rit[j];
        sinc(tsin, dim, res, s, dt, d[j], row, point[2]);
        int idx = find_t_min(t, tgrid, ntempi)+1;
        for (int k = 0; k < dim; k++)
            {im[idx - NL + k + m * 300] = im[idx - NL + k + m * 300] + s[k];}
    }
    

    free(d);
    free(tpixel);
    free(rit);
    free(t);
    }
}
}
