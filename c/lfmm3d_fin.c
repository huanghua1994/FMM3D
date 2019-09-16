#include "stdlib.h"
#include "stdio.h"
#include "math.h"
#include "lfmm3d_c.h"
#include "complex.h"
#include "cprini.h"

#include <omp.h>
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

void Laplace_3d_matvec_std(
    const double *coord0, const int ld0, const int n0,
    const double *coord1, const int ld1, const int n1,
    const double *x_in, double *x_out
)
{
    const double *x0 = coord0 + ld0 * 0;
    const double *y0 = coord0 + ld0 * 1;
    const double *z0 = coord0 + ld0 * 2;
    const double *x1 = coord1 + ld1 * 0;
    const double *y1 = coord1 + ld1 * 1;
    const double *z1 = coord1 + ld1 * 2;
    for (int i = 0; i < n0; i++)
    {
        const double x0_i = x0[i];
        const double y0_i = y0[i];
        const double z0_i = z0[i];
        double sum = 0.0;
        #pragma omp simd
        for (int j = 0; j < n1; j++)
        {
            double dx = x0_i - x1[j];
            double dy = y0_i - y1[j];
            double dz = z0_i - z1[j];
            double r2 = dx * dx + dy * dy + dz * dz;
            sum += (r2 == 0.0) ? 0.0 : (x_in[j] / sqrt(r2));
        }
        x_out[i] += sum;
    }
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


    printf("[INFO] lfmm3d_t_c_p: file coordinate input, source = target\n");
    lfmm3d_t_c_p_(&eps, &npoint, coord, charge, &npoint, coord, poten);
    for (int k = 0; k < 5; k++)
    {
        double st = get_wtime_sec();
        lfmm3d_t_c_p_(&eps, &npoint, coord, charge, &npoint, coord, poten);
        double et = get_wtime_sec();
    }
    
    double *coord1 = (double*) malloc(sizeof(double) * npoint * 3);
    double *poten1 = (double*) malloc(sizeof(double) * npoint);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < npoint; j++)
            coord1[i * npoint + j] = coord[j * 3 + i];
    memset(poten1, 0, sizeof(double) * npoint);
    
    int nthread = omp_get_max_threads();
    int ntrgchk = (npoint < 10000) ? npoint : 10000;
    #pragma omp parallel
    {
        int tid  = omp_get_thread_num();
        int sidx  = tid * ntrgchk / nthread;
        int eidx  = (tid + 1) * ntrgchk / nthread;
        int ntrgt = eidx - sidx;
        Laplace_3d_matvec_std(
            coord1 + sidx, npoint, ntrgt,
            coord1, npoint, npoint,
            charge, poten1 + sidx
        );
    }
    
    double std_l2 = 0, err_l2 = 0;
    for (int i = 0; i < ntrgchk; i++)
    {
        double diff = poten1[i] - poten[i];
        std_l2 += poten1[i] * poten1[i];
        err_l2 += diff * diff;
    }
    std_l2 = sqrt(std_l2);
    err_l2 = sqrt(err_l2);
    printf("Relative L2 error = %e\n", err_l2 / std_l2);
    
    free(coord1);
    free(coord);
    free(charge);
    free(poten);

    return 0;
}
