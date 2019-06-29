A = [-0.1005    1.0939    2.0428    4.4599   -1.0880   -0.1444    5.9859   -3.0481   -2.0510   -5.9709   -0.1387    1.9229   -4.4575    3.0753   -1.8847   -0.1164];
A = reshape(A, 4, 4)';

% (a)
eig(A)

%(b)
sys = ss(A, [], [], []);
x0 = [1; 1; 1; 1];
[y,t,x] = initial(sys, x0);
for i = 1: 4
    figure(i);
    plot(x(:, i));
end

x0 = rand(4, 1);
[y,t,x] = initial(sys, x0);
for i = 1: 4
    figure(i + 4);
    plot(x(:, i));
end

%(c)
expm(15 * A)

%(d)
expm(- 20 * A)

%(e)
x10 = [1; 1; 1; 1];
expm(10 * A) \ x10