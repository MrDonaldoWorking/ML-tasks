#include <stdio.h>
#include <vector>
#include <algorithm>

int main() {
    int k, n;
    scanf("%d%d", &k, &n);

    std::vector<std::vector<int>> classes(k);
    std::vector<std::pair<int, int>> pairs;
    std::vector<int> cnt(k);
    for (int i = 0; i < n; ++i) {
        int x, y;
        scanf("%d%d", &x, &y);

        classes[--y].push_back(x);
        pairs.push_back(std::make_pair(x, y));
        ++cnt[y];
    }

    for (int i = 0; i < k; ++i) {
        std::sort(classes[i].begin(), classes[i].end());
    }
    std::sort(pairs.begin(), pairs.end());

    long long intra = 0;
    for (int y = 0; y < k; ++y) {
        if (classes[y].empty()) {
            continue;
        }

        long long suffix = 0;
        for (int x : classes[y]) {
            suffix += x;
        }
        long long prefix = 0;
        size_t const size = classes[y].size();
        for (size_t i = 0; i < size; ++i) {
            int const curr = classes[y][i];
            suffix -= curr;
            intra += suffix - curr * (size - i - 1LL);
            prefix += curr;
            intra += curr * (i + 1LL) - prefix;
        }
    }

    long long all_suffix = 0, all_prefix = 0;
    std::vector<long long> suffixes(k), prefixes(k);
    for (auto const& p : pairs) {
        all_suffix += p.first;
        suffixes[p.second] += p.first;
    }
    // printf("all_suffix = %lld\n", all_suffix);
    // for (int i = 0; i < k; ++i) {
    //     printf("suffix[%d] = %lld\n", i, suffixes[i]);
    // }

    long long inter = 0;
    std::vector<int> curr_cnt(k);
    for (int i = 0; i < n; ++i) {
        int const x = pairs[i].first, y = pairs[i].second;
        ++curr_cnt[y];
        all_suffix -= x;
        suffixes[y] -= x;
        inter += all_suffix - suffixes[y];
        inter -= (n - i - 1LL - (cnt[y] - curr_cnt[y])) * x;
        all_prefix += x;
        prefixes[y] += x;
        inter += (i + 1LL - curr_cnt[y]) * x;
        inter -= all_prefix - prefixes[y];
    }

    printf("%lld\n%lld", intra, inter);

    return 0;
}
