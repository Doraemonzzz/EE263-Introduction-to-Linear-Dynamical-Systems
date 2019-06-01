G = [2 3; 1 0; 0 4; 1 1; -1 2];
G_tilde = [-3 -1; -1 0; 2 -3; -1 -3; 1 2];

A = [G'; G_tilde'];
b = [eye(2); eye(2)];

X = pinv(A) * b;
h = X';

%验证结果
h * G
h * G_tilde