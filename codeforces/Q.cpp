#include <stdio.h>
#include <vector>
#include <map>
#include <math.h>

int main() {
    int kx, ky, n;
    scanf("%d%d%d", &kx, &ky, &n);

    std::vector<int> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        scanf("%d%d", &x[i], &y[i]);
    }

    std::vector<double> prob_x(kx);
    for (int i = 0; i < n; ++i) {
        ++prob_x[x[i] - 1];
    }
    for (int i = 0; i < kx; ++i) {
        prob_x[i] /= n;
    }

    std::map<std::pair<int, int>, double> prob_xy;
    for (int i = 0; i < n; ++i) {
        ++prob_xy[std::make_pair(x[i] - 1, y[i] - 1)];
    }
    for (auto &elem : prob_xy) {
        elem.second /= n;
    }

    double res = 0;
    for (auto const& elem : prob_xy) {
        res -= elem.second * log(elem.second / prob_x[elem.first.first]);
    }

    printf("%.10f", res);

    return 0;
} 
