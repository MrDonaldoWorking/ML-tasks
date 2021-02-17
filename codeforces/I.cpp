#include <stdio.h>
#include <vector>
#include <math.h>
#include <string>

struct matrix {
    int n, m;
    std::vector<std::vector<double>> a;

    matrix() {}

    matrix(int const x, int const y): n(x), m(y) {
        a.resize(x, std::vector<double>(y));
    }

    matrix operator *(matrix const& other) const {
        matrix res(n, other.m);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < other.m; ++j) {
                double curr = 0;
                for (int k = 0; k < m; ++k) {
                    curr += a[i][k] * other.a[k][j];
                }
                res.a[i][j] = curr;
            }
        }
        return res;
    }

    matrix operator +(matrix const& other) const {
        matrix res(n, m);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                res.a[i][j] = a[i][j] + other.a[i][j];
            }
        }
        return res;
    }
};

matrix transpose(matrix const& m) {
    matrix t(m.m, m.n);
    for (int i = 0; i < t.n; ++i) {
        for (int j = 0; j < t.m; ++j) {
            t.a[i][j] = m.a[j][i];
        }
    }
    return t;
}

void print_matrix(matrix const& m) {
    for (int i = 0; i < m.n; ++i) {
        for (int j = 0; j < m.m; ++j) {
            printf("%.10f ", m.a[i][j]);
        }
        printf("\n");
    }
}

struct vertex {
    matrix m, bm, fm;

    virtual void fore(std::vector<vertex *> const&) = 0;
    virtual void back(std::vector<vertex *> &) = 0;

    virtual ~vertex() {}
};

struct var : vertex {
    var(int const r, int const c) {
        m = matrix(r, c);
        // printf("entered var %d %d\n", m.n, m.m);
    }
    void fore(std::vector<vertex *> const& v) override {
        bm = matrix(m.n, m.m);
    }
    void back(std::vector<vertex *> & v) override {
        fm = bm;
    }
};

struct tnh : vertex {
    tnh(int const i): index(i) {}

    void fore(std::vector<vertex *> const& vertices) override {
        m = matrix(vertices[index]->m.n, vertices[index]->m.m);
        bm = matrix(m.n, m.m);
        fm = matrix(m.n, m.m);
        for (int i = 0; i < m.n; ++i) {
            for (int j = 0; j < m.m; ++j) {
                m.a[i][j] = tanh(vertices[index]->m.a[i][j]);
            }
        }
    }

    void back(std::vector<vertex *> &vertices) override {
        // bm is initialized
        for (int i = 0; i < bm.n; ++i) {
            for (int j = 0; j < bm.m; ++j) {
                fm.a[i][j] = bm.a[i][j] * (1 - m.a[i][j] * m.a[i][j]);
                vertices[index]->bm.a[i][j] += fm.a[i][j];
            }
        }
    }
 private:
    int const index;
};

struct rlu : vertex {
    rlu(double const alpha, int const ind): a(1 / alpha), index(ind) {}

    void fore(std::vector<vertex *> const& vertices) override {
        m = matrix(vertices[index]->m.n, vertices[index]->m.m);
        bm = matrix(m.n, m.m);
        fm = matrix(m.n, m.m);
        for (int i = 0; i < m.n; ++i) {
            for (int j = 0; j < m.m; ++j) {
                m.a[i][j] = vertices[index]->m.a[i][j];
                if (vertices[index]->m.a[i][j] < 0) {
                    m.a[i][j] *= a;
                }
            }
        }
    }

    void back(std::vector<vertex *> &vertices) override {
        for (int i = 0; i < m.n; ++i) {
            for (int j = 0; j < m.m; ++j) {
                fm.a[i][j] = bm.a[i][j];
                if (m.a[i][j] < 0) {
                    fm.a[i][j] *= a;
                }
                vertices[index]->bm.a[i][j] += fm.a[i][j];
            }
        }
    }
 private:
    double const a;
    int const index;
};

struct mul : vertex {
    mul(int const x, int const y): index1(x), index2(y) {
        // printf("mul %d %d\n", x, y);
    }

    void fore(std::vector<vertex *> const& vertices) override {
        m = vertices[index1]->m * vertices[index2]->m;
        bm = matrix(m.n, m.m);
        fm = matrix(m.n, m.m);
    }

    void back(std::vector<vertex *> &vertices) override {
        fm = bm;
        vertices[index1]->bm = vertices[index1]->bm + fm * transpose(vertices[index2]->m);
        vertices[index2]->bm = vertices[index2]->bm + transpose(vertices[index1]->m) * fm;
    }

 private:
    int const index1, index2;
};

struct sum : vertex {
    sum(std::vector<int> const& ind): indices(ind) {}

    void fore(std::vector<vertex *> const& vertices) override {
        m = matrix(vertices[indices[0]]->m.n, vertices[indices[0]]->m.m);
        for (int ind : indices) {
            m = m + vertices[ind]->m;
        }
        bm = matrix(m.n, m.m);
        fm = matrix(m.n, m.m);
    }

    void back(std::vector<vertex *> &vertices) override {
        // printf("sum back\n");
        fm = bm;
        // printf("fm = bm:\n");
        // print_matrix(fm);
        for (int ind : indices) {
            // printf("vertices[%d]-> += fm\n", ind);
            // printf("vertices[%d]->bm is:\n", ind);
            // print_matrix(vertices[ind]->bm);
            vertices[ind]->bm = vertices[ind]->bm + fm;
        }
    }

 private:
    std::vector<int> indices;
};

struct had : vertex {
    had(std::vector<int> const& ind): indices(ind) {}

    void fore(std::vector<vertex *> const& vertices) override {
        int const x = vertices[indices[0]]->m.n;
        int const y = vertices[indices[0]]->m.m;
        m = matrix(x, y);
        bm = matrix(x, y);
        fm = matrix(x, y);
        for (int i = 0; i < x; ++i) {
            for (int j = 0; j < y; ++j) {
                m.a[i][j] = 1;
            }
        }
        for (int ind : indices) {
            for (int i = 0; i < x; ++i) {
                for (int j = 0; j < y; ++j) {
                    m.a[i][j] *= vertices[ind]->m.a[i][j];
                }
            }
        }
    }

    void back(std::vector<vertex *> &vertices) override {
        for (int i = 0; i < m.n; ++i) {
            for (int j = 0; j < m.m; ++j) {
                for (size_t k = 0; k < indices.size(); ++k) {
                    double mul = 1;
                    for (size_t l = 0; l < indices.size(); ++l) {
                        if (k == l) {
                            continue;
                        }
                        mul *= vertices[indices[l]]->m.a[i][j];
                    }
                    fm.a[i][j] = mul * bm.a[i][j];
                    vertices[indices[k]]->bm.a[i][j] += fm.a[i][j];
                }
            }
        }
    }

 private:
    std::vector<int> indices;
};

bool read_str(std::string &s) {
    char c;
    while (scanf("%c", &c) == 1) {
        if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) {
            break;
        }
    }
    s += c;
    while (scanf("%c", &c) == 1) {
        if (c == ' ' || c == '\n') {
            break;
        }

        s += c;
    }
    // printf("s[0] = %c\n", s[0]);

    return !s.empty();
}

int main() {
    int n, m, k;
    scanf("%d%d%d", &n, &m, &k);

    std::vector<vertex *> vertices;
    for (int i = 0; i < n; ++i) {
        std::string s;
        read_str(s);

        int a, b;
        std::vector<int> indices;
        switch (s[0]) {
            case 'v':
                scanf("%d%d", &a, &b);
                vertices.push_back(new var(a, b));
                break;

            case 't':
                scanf("%d", &a);
                vertices.push_back(new tnh(--a));
                break;

            case 'r':
                scanf("%d%d", &a, &b);
                vertices.push_back(new rlu(a, --b));
                break;

            case 'm':
                scanf("%d%d", &a, &b);
                // printf("mul, scanned %d %d\n", a, b);
                vertices.push_back(new mul(--a, --b));
                break;
            
            case 's':
                scanf("%d", &a);
                while (a--) {
                    int x;
                    scanf("%d", &x);
                    indices.push_back(--x);
                }
                vertices.push_back(new sum(indices));
                break;

            case 'h':
                scanf("%d", &a);
                while (a--) {
                    int x;
                    scanf("%d", &x);
                    indices.push_back(--x);
                }
                vertices.push_back(new had(indices));
                break;
        }
    }

    for (int ind = 0; ind < m; ++ind) {
        for (int i = 0; i < vertices[ind]->m.n; ++i) {
            for (int j = 0; j < vertices[ind]->m.m; ++j) {
                scanf("%lf", &vertices[ind]->m.a[i][j]);
            }
        }
        // printf("end of %d\n", ind);
    }

    for (int ind = 0; ind < n; ++ind) {
        vertices[ind]->fore(vertices);
    }

    for (int ind = 0; ind < k; ++ind) {
        print_matrix(vertices[n - k + ind]->m);
    }

    for (int ind = 0; ind < k; ++ind) {
        int const ii = n - k + ind;
        for (int i = 0; i < vertices[ii]->m.n; ++i) {
            for (int j = 0; j < vertices[ii]->m.m; ++j) {
                scanf("%lf", &vertices[ii]->bm.a[i][j]);
            }
        }
    }

    // printf("going another way\n");
    for (int ind = n - 1; ind > -1; --ind) {
        // printf("ind = %d\n", ind);
        vertices[ind]->back(vertices);
    }

    for (int ind = 0; ind < m; ++ind) {
        print_matrix(vertices[ind]->fm);
    }

    for (int ind = 0; ind < n; ++ind) {
        delete vertices[ind];
    }

    return 0;
} 
