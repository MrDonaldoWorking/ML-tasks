#include <stdio.h>
#include <vector>
#include <map>
#include <math.h>

int main() {
    int n;
    scanf("%d", &n);

    std::vector<int> x1(n), x2(n);
    double x1_mean = 0, x2_mean = 0;
    for (int i = 0; i < n; ++i) {
        scanf("%d%d", &x1[i], &x2[i]);
        x1_mean += x1[i];
        x2_mean += x2[i];
    }
    x1_mean /= n;
    x2_mean /= n;

    double cov = 0;
    for (int i = 0; i < n; ++i) {
        cov += (x1[i] - x1_mean) * (x2[i] - x2_mean);
    }

    double sx1 = 0, sx2 = 0;
    for (int i = 0; i < n; ++i) {
        sx1 += (x1[i] - x1_mean) * (x1[i] - x1_mean);
        sx2 += (x2[i] - x2_mean) * (x2[i] - x2_mean);
    }
    
    double const EPS = 1e-6;
    if (sx1 * sx2 <= EPS) {
        printf("0");
    } else {
        printf("%.10f", cov / sqrt(sx1 * sx2));
    }

    return 0;
} 
