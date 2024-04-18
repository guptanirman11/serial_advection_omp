#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int main(int argc, char *argv[]){
    if (argc != 8){
        printf("Usage: %s N NT L T u v\n", argv[0]);
        return 1;
    }
    
    int N = atoi(argv[1]);
    int NT = atoi(argv[2]);
    double L = atof(argv[3]);
    double T = atof(argv[4]);
    double u = atof(argv[5]);
    double v = atof(argv[6]);
    int nt = atoi(argv[7]);

    double dx = L/N;
    double dt = T/NT;
    double dy = dx;

    omp_set_num_threads(nt); // Setting the number of threads

    // Ensuring Courant stability condition
    if (dt > dx/ (sqrt(2*(u*u + v*v)))){
        printf("Courant condition not satisfied\n");
        return 1;
    }

    // Allocate memory for concentration arrays
    double **C = (double **)malloc(N*sizeof(double *));
    double **C_new = (double **)malloc(N*sizeof(double *));


    for (int i = 0; i < N; i++){
        C[i] = (double *)malloc(N*sizeof(double));
        C_new[i] = (double *)malloc(N*sizeof(double));
    }

   
    double x_0 = L/2;
    double y_0 = L/2;
    double sigma = L/4;

    #pragma omp parallel for default(none) shared(C, N, dx, dy, x_0, y_0, sigma) private(i, j)
    // Initialize concentration array
    for (int i = 0;i < N; i++){
        for (int j = 0; j<N; j++){
            double x = i*dx;
            double y = j*dy;
            C[i][j] = exp(-((x-x_0)*(x-x_0) + (y-y_0)*(y-y_0))/(2*sigma*sigma));
        }
    }

    // Time loop and boundary conditions
    for (int t = 0; t < NT; t++){
        for (int i = 0; i < N; i++){
            for (int j = 0; j < N; j++){
                // Have to calculate first and then apply boundary conditions
                int im1 = (i == 0) ? N-1 : i-1;
                int ip1 = (i == N-1) ? 0 : i+1;
                int jm1 = (j == 0) ? N-1 : j-1;
                int jp1 = (j == N-1) ? 0 : j+1;

                C_new[i][j] = 0.25 * (C[im1][j] + C[ip1][j] + C[i][jm1] + C[i][jp1])
                            - dt / (2 * dx) * (u * (C[ip1][j] - C[im1][j]) + v * (C[i][jp1] - C[i][jm1]));
        
            }
        
        }
        // Applying Boundary conditions
        for (int i = 0; i < N; i++){
            for (int j = 0; j < N; j++){
                if (i == 0){
                    C_new[i][j] = C_new[N-1][j];
                }
                if (i == N-1){
                    C_new[i][j] = C_new[0][j];
                }
                if (j == 0){
                    C_new[i][j] = C_new[i][N-1];
                }
                if (j == N-1){
                    C_new[i][j] = C_new[i][0];
                }
            }

        }
        // Swap pointers
        double **tmp = C;
        C = C_new;
        C_new = tmp;
    }
    // Free memory
    for (int i = 0; i < N; i++){
        free(C[i]);
        free(C_new[i]);
    }

    free(C);
    free(C_new);
    
    return 0;

}