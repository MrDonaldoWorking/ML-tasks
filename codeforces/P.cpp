#include <stdio.h>
#include <vector>
#include <map>
#include <algorithm>

int main() {
    int k1, k2, n;
    scanf("%d%d%d", &k1, &k2, &n);

    std::vector<int> x1(n), x2(n);
    for (int i = 0; i < n; ++i) {
        scanf("%d%d", &x1[i], &x2[i]);
    }

    std::vector<double> mean_x1(k1), mean_x2(k2);
    for (int i = 0; i < n; ++i) {
        ++mean_x1[x1[i] - 1];
        ++mean_x2[x2[i] - 1];
    }
    for (int i = 0; i < k1; ++i) {
        mean_x1[i] /= n;
    }
    for (int i = 0; i < k2; ++i) {
        mean_x2[i] /= n;
    }

    std::map<std::pair<int, int>, int> cnt;
    for (int i = 0; i < n; ++i) {
        ++cnt[std::make_pair(x1[i] - 1, x2[i] - 1)];
    }

    double res = n;
    for (auto const& elem : cnt) {
        int const x1_v = elem.first.first, x2_v = elem.first.second;
        int const v = elem.second;
        
        double const m = n * mean_x1[x1_v] * mean_x2[x2_v];
        double const p = (v - m) * (v - m) / m;
        res = res - m + p;
    }

    printf("%.10f", res);

    return 0;
} 
