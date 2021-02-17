#include <stdio.h>
#include <vector>
#include <algorithm>

int main() {
    int k, n;
    scanf("%d%d", &k, &n);

    std::vector<int> x, y;
    for (int i = 0; i < n; ++i) {
        int curr_x, curr_y;
        scanf("%d%d", &curr_x, &curr_y);

        x.push_back(curr_x);
        y.push_back(curr_y);
    }

    std::vector<double> x_prob(k);
    for (int i = 0; i < n; ++i) {
        x_prob[x[i] - 1] += 1.0 / n;
    }

    double mean_y2 = 0;
    for (int i = 0; i < n; ++i) {
        mean_y2 += (double) y[i] * y[i] / n;
    }

    std::vector<double> y_by_x(k);
    for (int i = 0; i < n; ++i) {
        y_by_x[x[i] - 1] += (double) y[i] / n;
    }

    double mean_yx2 = 0;
    double const EPS = 1e-6;
    for (int i = 0; i < k; ++i) {
        if (x_prob[i] < EPS) {
            continue;
        }
        mean_yx2 += y_by_x[i] * y_by_x[i] / x_prob[i];
    }

    printf("%.10f", mean_y2 - mean_yx2);

    return 0;
}
