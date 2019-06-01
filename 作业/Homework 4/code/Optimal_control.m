% (a)
a1 = linspace(9.5, 0.5, 10);
a2 = ones(1,10);
a3 = [linspace(4.5, 0.5, 5), 0 0 0 0 0];

A = [a1; a2; a3];
b = [1; 0; 0];
x = pinv(A) * b

% ×÷Í¼
% f(t)
figure(1);
subplot(3,1,1);
stairs(0:10, [x; 0]);
grid on;
xlabel('t');
ylabel('f(t)');
axis([0, 10, min(x) - 0.1, max(x) + 0.1]);

% p_dot(t)
T1 = toeplitz(ones(10,1), [1,zeros(1,9)]);
p_dot = T1 * x;
subplot(3,1,2);
plot(linspace(0, 10, 10), p_dot);
grid on;
xlabel('t');
ylabel('p_{dot}(t)');
axis([0, 10, min(p_dot) - 0.1, max(p_dot) + 0.1]);

% p(t)
T2 = toeplitz(linspace(0.5, 9.5, 10)', [0.5, zeros(1,9)]);
p = T2 * x;
subplot(3,1,3);
plot(linspace(0, 10, 10), p);
grid on;
xlabel('t');
ylabel('p(t)');
axis([0, 10, min(p) - 0.1, max(p) + 0.1]);

% (b)
% ×÷Í¼
N = 50;
[d, m] = size(A);
Mu = logspace(-5, 2, N);
J1 = zeros(N, 1);
J2 = zeros(N, 1);

for i = 1:N
    mu = Mu(i);
    %x = inv(A' * A + mu * eye(m))  * A' * b;
    x = (A' * A + mu * eye(m)) \ (A' * b);
    j1 = norm(A * x - b) ^ 2;
    j2 = norm(x) ^ 2;
    J1(i) = j1;
    J2(i) = j2;
end

figure(2);
plot(J1, J2);
axis tight;