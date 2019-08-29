A = [1, 0, 0, 0; 1, 1, 0, 0; 0, 1, 1, 0; 1, 0, 0, 0];
B = [0, 1; 0, 1; 1, 0; 0, 0];
x0 = [1; 0; -1; 1];
T = 1;
C = B;
tmp = B;
x = A * x0;

while true
    if rank(C) == rank([C, x])
        break
    end
    x = A * x;
    tmp = A * tmp;
    C = [C, tmp];
    T = T + 1;
end

% (a)
u = - pinv(C) * x;
J1 = norm(u) ^ 2

% (b)
C = [];
tmp = B;
x10 = x0;
for i = 1:10
    C = [C, tmp];
    tmp = A * tmp;
    x10 = A * x10;
end

P = C;
v = - x10;
[m, n] = size(P);
N = 100;
Lambda = logspace(1, -1, N);
res = zeros(1, N);
for i = 1: N
    lambda = Lambda(i);
    u = inv(eye(n) + lambda * P' * P) * lambda * P' * v;
    res(i) = norm(P * u - v) - 0.1;
    if i > 1 && res(i) * res(i - 1) < 0
        u_res = u;
        %break;
    end
end

J9 = norm(u_res) ^ 2;
plot(res);
