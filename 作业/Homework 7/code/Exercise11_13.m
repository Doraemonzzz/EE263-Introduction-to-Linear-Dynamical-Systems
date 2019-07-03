N = 10;

while 1
    A = randn(N);
    %����ֵ�ֽ�
    [S0, Lambda] = eig(A);
    Lambda = diag(Lambda);
    %���㸴����ֵ������
    index = imag(Lambda) ~= 0;
    if sum(index) > 0
        break
    end
end

S = zeros(N);
i = 1;
while i <= N
    if index(i)
        S(:, i) = real(S0(:, i));
        S(:, i + 1) = imag(S0(:, i));
        i = i + 2;
    else
        S(:, i) = S0(:, i);
        i = i + 1;
    end
end

S \ A * S
