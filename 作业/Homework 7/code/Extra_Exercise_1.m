n = 3;
A = randn(n);
[S1, Lambda1] = eig(A);
Lambda1 = diag(Lambda1);

% method 1
B = (eye(n) + A) / (eye(n) - A);
[S2, Lambda2] = eig(B);
Lambda2 = diag(Lambda2);

% method 2
Lambda3 = (1 + Lambda1) ./ (1 - Lambda1);

Lambda2
Lambda3