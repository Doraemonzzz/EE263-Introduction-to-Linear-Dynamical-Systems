% EE263 2001 Final.% File creates the matrix A, which has dimensions NxN.N=5;                        % Set dimensions of AVmax=3;                     % Set max noise amplitude% Generate matrix A
A = [2 4 5 4 5;    0 5 7 7 1;    7 8 0 6 7;     7 0 4 9 4;    9 1 1 8 7];
Vmax = 3;
[U, S, V] = svd(A);
k = Vmax / S(1, 1);
s1 = k * V(:, 1)
s2 = - k * V(:, 1)