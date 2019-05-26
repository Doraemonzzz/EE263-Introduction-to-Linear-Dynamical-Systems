N = 1000;
R1 = zeros(N, 1);
R2 = zeros(N, 1);

for i = 1:N
    % (a)生成数据
    A = randn(50, 20);
    v = 0.1 * randn(50, 1);
    x = randn(20, 1);
    y = A * x + v;

    % (b)最小二乘
    xls = A \ y;
    r1 = norm(xls - x) / norm(x);

    % (c)
    y_trunc = y(1:20, :);
    A_trunc = A(1:20, :);
    xjem = A_trunc \ y_trunc;
    r2 = norm(xjem - x) / norm(x);
    
    R1(i) = r1;
    R2(i) = r2;
end

mean(R1)
mean(R2)