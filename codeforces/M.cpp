#include <stdio.h>
#include <vector>
#include <algorithm>

int main() {
    int n;
    scanf("%d", &n);

    std::vector<std::pair<int, int>> x1, x2;
    for (int i = 0; i < n; ++i) {
        int curr_x1, curr_x2;
        scanf("%d%d", &curr_x1, &curr_x2);
        x1.push_back(std::make_pair(curr_x1, i));
        x2.push_back(std::make_pair(curr_x2, i));
    }
    std::sort(x1.begin(), x1.end());
    std::sort(x2.begin(), x2.end());

    std::vector<int> num_x1(n), num_x2(n);
    for (int i = 0; i < n; ++i) {
        num_x1[x1[i].second] = i;
        num_x2[x2[i].second] = i;
    }

    double rank_sum = 0;
    for (int i = 0; i < n; ++i) {
        double const diff = num_x1[i] - num_x2[i];
        rank_sum += diff * diff;
    }

    printf("%.10f", 1 - 6.0 / ((double) n * (n + 1) * (n - 1)) * rank_sum);

    return 0;
} 
