A = zeros(N, n_pixels^2);
for i = 1 : N
    data = line_pixel_length(lines_d(i),lines_theta(i),n_pixels);
    data = data(:);
    A(i, :) = data;
end

% v = inv(A' * A) * A' * y;
v = A \ y;
X = reshape(v, n_pixels, n_pixels);
figure(1)      % display the original image
colormap gray
imagesc(X)
axis image