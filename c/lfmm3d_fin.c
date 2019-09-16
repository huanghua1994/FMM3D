#include "stdlib.h"
#include "stdio.h"
#include "math.h"
#include "lfmm3d_c.h"
#include "complex.h"
#include "cprini.h"

#include <string.h>
#include <sys/time.h>

double get_wtime_sec()
{
    double sec;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    sec = tv.tv_sec + (double) tv.tv_usec / 1000000.0;
    return sec;
}

int main(int argc, char **argv)
{
    cprin_init("stdout", "fort.13");
    cprin_skipline(2);

    int npoint = atoi(argv[1]);
    double eps = atof(argv[2]);
    printf("Number of points = %d, eps = %e\n", npoint, eps);
    
    double *coord  = (double*) malloc(sizeof(double) * npoint * 3);
    double *charge = (double*) malloc(sizeof(double) * npoint);
    double *poten  = (double*) malloc(sizeof(double) * npoint);
    int need_gen = 1;
    if (argc >= 3)
    {
        if (strstr(argv[3], ".csv") != NULL)
        {
            printf("Reading coordinates from CSV file...");
            FILE *inf = fopen(argv[3], "r");
            for (int i = 0; i < npoint; i++)
            {
                fscanf(inf, "%lf,%lf,%lf\n", &coord[3 * i], &coord[3 * i + 1], &coord[3 * i + 2]);
            }
            fclose(inf);
            printf(" done.\n");
            need_gen = 0;
        }
        if (strstr(argv[3], ".bin") != NULL)
        {
            printf("Reading coordinates from binary file...");
            FILE *inf = fopen(argv[3], "rb");
            fread(coord, sizeof(double), npoint * 3, inf);
            fclose(inf);
            printf(" done.\n");
            need_gen = 0;
        }
    }
    if (need_gen == 1)
    {
        printf("Binary/CSV coordinate file not provided. Generating random coordinates in unit box...");
        for (int i = 0; i < 3 * npoint; i++) coord[i] = rand01();
        printf(" done.\n");
    }
    for (int i = 0; i < npoint; i++) charge[i] = rand01();


    cprin_message("[INFO] lfmm3d_t_c_p: file coordinate input, source = target");
    cprin_skipline(2);

    lfmm3d_t_c_p_(&eps, &npoint, coord, charge, &npoint, coord, poten);
    for (int k = 0; k < 5; k++)
    {
        double st = get_wtime_sec();
        lfmm3d_t_c_p_(&eps, &npoint, coord, charge, &npoint, coord, poten);
        double et = get_wtime_sec();
    }

    free(coord);
    free(charge);
    free(poten);

    return 0;
}
