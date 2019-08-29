c = [  0.0455; -0.2273; -0.0455;  0.2727;  0.4545;  0.4545;  0.2727; -0.0455; -0.2273;  0.0455;];
k = 1;
n = length(c);
A = zeros(n, n);
B = zeros(n - 1, n);
for i = 1: n
    for j = 1:i
        A(i, j) = c(i + 1 - j)
    end
end

for i = 1: (n - 1)
    for j = 1 : (n - i)
        B(i, j + i) = c(n + 1 - j)
    end
end

C = [A; B];
D = [zeros(2 * k + 1, n - k - 1), eye(2 * k + 1), zeros(2 * k + 1, n - k - 1)];
%E = inv(C' * C) * (C' * D' * D * C);
E = (C' * C) \ (C' * D' * D * C);
Eig = eig(E);
res = max(Eig)

