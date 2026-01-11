#include <bits/stdc++.h>

using namespace std;
using ll = long long;
const ll INF = 4e18;

long double calc_det(vector<vector<long double>>& A) {
    ll n = A.size();
    long double det = 1;
    long double eps = 1e-12;
    for (int i = 0; i < n; ++i) {
        int k = i;
        for (int j = i + 1; j < n; ++j)
            if (abs(A[j][i]) > abs(A[k][i]))
                k = j;
        if (abs(A[k][i]) < eps) {
            det = 0;
            break;
        }
        swap(A[i], A[k]);
        if (i != k)
            det = -det;
        det *= A[i][i];
        for (int j = i + 1; j < n; ++j) {
            A[i][j] /= A[i][i];
        }
        for (int j = 0; j < n; ++j) {
            if (j != i && abs(A[j][i]) > eps) {
                for (int k = i + 1; k < n; ++k) {
                    A[j][k] -= A[i][k] * A[j][i];
                }
            }
        }
    }
    return det;
}
bool is_sym(vector<vector<long double>>& A) {
    const double eps = 1e-12;
    for (int i = 0; i < A.size(); i++) {
        for (int j = 0; j < A[i].size(); j++) {
            if (abs(A[i][j] - A[j][i]) > eps) {
                return false;
            }
        }
    }
    return true;
}
bool check_silv(vector<vector<long double>>& A) {
    ll n = A.size();
    const double eps = 1e-12;
    for (int i = 0; i < n; i++) {
        vector<vector<long double>> A_h(i + 1, vector<long double>(i + 1));
        for (int j = 0; j <= i; j++) {
            for (int k = 0; k <= i; k++) {
                A_h[j][k] = A[j][k];
            }
        }
        if (calc_det(A_h) < eps) {
            return false;
        }
    }
    return true;
}
vector<vector<long double>> gen_good(int n, int maxAbs, long double lambda = 1e-3) {
    mt19937_64 rng(time(0));
    uniform_int_distribution<int> dist(-maxAbs, maxAbs);

    vector<vector<long double>> M(n, vector<long double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            M[i][j] = (long double)dist(rng);
    vector<vector<long double>> A(n, vector<long double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            double s = 0.0;
            for (int k = 0; k < n; ++k) s += M[k][i] * M[k][j];
            A[i][j] = s;
            A[j][i] = s;
        }
    }
    for (int i = 0; i < n; ++i) A[i][i] += lambda;

    return A;
}
vector<long double> operator-(const vector<long double>& a, const vector<long double>& b) {
    vector<long double> c(a.size());
    for (int i = 0; i < a.size(); ++i) c[i] = a[i] - b[i];
    return c;
}
vector<long double> operator+(const vector<long double>& a, const vector<long double>& b) {
    vector<long double> c(a.size());
    for (int i = 0; i < a.size(); ++i) c[i] = a[i] + b[i];
    return c;
}
vector<long double> operator-(const vector<long double>& a) {
    vector<long double> c(a.size());
    for (size_t i = 0; i < a.size(); ++i) c[i] = -a[i];
    return c;
}
long double operator*(const vector<long double>& a, const vector<long double>& b) {
    long double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}
vector<long double> operator*(const vector<vector<long double>>& A, const vector<long double>& x) {
    const int n = A.size();
    const int m = A[0].size();
    vector<long double> y(n);
    for (int i = 0; i < n; ++i) {
        long double s = 0.0;
        for (int j = 0; j < m; ++j) s += A[i][j] * x[j];
        y[i] = s;
    }
    return y;
}
vector<long double> operator*(const vector<long double>& x, const vector<vector<long double>>& A) {
    const int n = A.size();
    const int m = A[0].size();

    vector<long double> y(m, 0.0);
    for (int j = 0; j < m; ++j) {
        long double s = 0.0;
        for (int i = 0; i < n; ++i) s += x[i] * A[i][j];
        y[j] = s;
    }
    return y;
}
vector<long double> operator*(const vector<long double>& a, long double& k) {
    vector<long double> r(a.size());
    for (int i = 0; i < a.size(); ++i) r[i] = a[i] * k;
    return r;
}
void print_ver(vector<long double>& x) {
    ll n = x.size();
    for (int i = 0; i < n; i++) {
        cout << fixed << setprecision(10) << x[i];
        cout << (i == n - 1 ? "" : ", ");
    }
    cout << '\n';
}
void print_ver(vector<long double>& x, ofstream &fout) {
    ll n = x.size();
    for (int i = 0; i < n; i++) {
        fout << fixed << setprecision(10) << x[i];
        fout << (i == n - 1 ? "" : ", ");
    }
    fout << '\n';
}
vector<long double> exac_solution(vector<vector<long double>>& A, vector<long double>& b) {
    int n = A.size();
    const long double eps = 1e-10;

    vector<vector<long double>> a = A;
    vector<long double> rhs = b;

    for (int col = 0; col < n; ++col) {
        int pivot = col;
        for (int row = col + 1; row < n; ++row) {
            if (fabsl(a[row][col]) > fabsl(a[pivot][col]))
                pivot = row;
        }

        if (fabsl(a[pivot][col]) < eps) {
            return vector<long double>();
        }

        if (pivot != col) {
            swap(a[pivot], a[col]);
            swap(rhs[pivot], rhs[col]);
        }

        for (int row = col + 1; row < n; ++row) {
            long double factor = a[row][col] / a[col][col];
            if (fabsl(factor) < eps) continue;
            a[row][col] = 0.0L;
            for (int j = col + 1; j < n; ++j) {
                a[row][j] -= factor * a[col][j];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    vector<long double> x(n, 0.0L);
    for (int i = n - 1; i >= 0; --i) {
        long double sum = rhs[i];
        for (int j = i + 1; j < n; ++j) {
            sum -= a[i][j] * x[j];
        }
        if (fabsl(a[i][i]) < eps) {
            return vector<long double>();
        }
        x[i] = sum / a[i][i];
    }

    return x;
}
long double F(vector<vector<long double>>& A, vector<long double>& b, vector<long double>& x) {
    return (x * A * x) / 2 - b * x;
}
long double L2_norm(vector<long double> x) {
    long double res = 0.0;
    for (int i = 0; i < x.size(); ++i)
        res += sqrt(x[i] * x[i]);
    return res;
}
long double L1_norm(vector<long double> x) {
    long double res = 0.0;
    for (int i = 0; i < x.size(); ++i)
        res += abs(x[i]);
    return res;
}
void solve() {
    vector<long double> x0;
    vector<vector<long double>> A;
    cout << "Enter size of matrix A\n";
    ll n;
    cin >> n;
    x0.resize(n);
    cout << "1. Enter matrix A by hands\n";
    cout << "2. Generate matrix A\n";
    ll type;
    cin >> type;
    A.resize(n);
    for (int i = 0; i < n; i++) {
        A[i].resize(n);
    }
    if (type == 1) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                cin >> A[i][j];
            }
        }
    }
    else {
        A = gen_good(n, 5, 1e-3);
    }
	if (!is_sym(A) || !check_silv(A)) {
        cout << "Bad matrix!\n";
        return;
    }
    cout << "Enter vector b\n";
    vector<long double> b(n);
    for (int i = 0; i < n; i++) {
        cin >> b[i];
    }
    
	ofstream fout_L2_error("L2error.txt");
	ofstream fout_log_L2_error("log_L2error.txt");
	
	vector<long double> exact_solution = exac_solution(A, b);	
    vector<long double> r = A * x0 - b;
    vector<long double> p = -r;
	
    for (int i = 0; i < n; i++) {
        long double alph = (-r * p) / (p * A * p);
        vector<long double> nxt_x = x0 + (p * alph);
        vector<long double> nxt_r = A * nxt_x - b;
        long double bth = (nxt_r * A * p) / (p * A * p);
        vector<long double> nxt_p = -nxt_r + (p * bth);

        swap(p, nxt_p);
        swap(nxt_x, x0);
        swap(nxt_r, r);
        vector<long double>().swap(nxt_r);
        vector<long double>().swap(nxt_x);
        vector<long double>().swap(nxt_p);
		
		fout_L2_error << i + 1 << ' ' << fixed << setprecision(8) << L2_norm(x0 - exact_solution) << '\n';
		fout_log_L2_error << i + 1 << ' ' << fixed << setprecision(8) << log(L2_norm(x0 - exact_solution)) << '\n';
		
        // cout << "Step " << i + 1 << '\n';
        // cout << "Intermediate point \n";
        // print_ver(x0);
        // cout << "Intermediate value\n";
        // cout << F(A, b, x0) << '\n';
    }
	fout_L2_error.close();
	fout_log_L2_error.close();
	
	ofstream fout("output.txt");
	
    fout << "Founded minimum point\n";
    print_ver(x0, fout);
	
    fout << "Founded minimum value\n";
    fout << F(A, b, x0) << '\n';
	
	fout << "Exact solution\n";
    print_ver(exact_solution, fout);
	
    fout << "Exact minimum value\n";
    fout << F(A, b, exact_solution) << '\n';
	
	fout.close();
}
int main() {
    int t;
    t = 1;
	//cin >> t;
    while (t--) {
        solve();
    }
    return 0;
}