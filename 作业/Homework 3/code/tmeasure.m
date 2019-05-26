% file: tmeasure.m
%
% image reconstruction from line measurements (tomography via least-squares)
%
% this is the matlab script that was used to produce the data in tomodata.m

n_pixels=30;   % image size is n_pixels by n_pixels
Nd=35;        % number of parallel lines for each angle
Ntheta=35;    % number of angles (equally spaced from 0 to pi)
N=Nd*Ntheta;  % total number of lines (i.e. number of measurements

sigma=0.7;   % noise level (standard deviation for normal dist.)

X=  % ...the mystery image goes here... (we hid this part!)

x=X(:);  % write the matrix X (the original image) as one big column vector
         % (first column of X goes first, then 2nd column, etc.)

figure(1)      % display the original image
colormap gray
imagesc(X)
axis image

y=zeros(N,1);            % will store the N measurements
lines_d=zeros(N,1);      % will store the position of each line
lines_theta=zeros(N,1);  % will store the angle of each line
i=1;
for itheta=1:Ntheta
 for id=1:Nd
  lines_d(i)=0.7*n_pixels*(id-Nd/2-0.5)/(Nd/2);
           % equally spaced parallel lines, distance from first to
           % last is about 1.4*n_pixels (to ensure coverage of whole
           % image when theta=pi/4)
  lines_theta(i)=pi*itheta/Ntheta;
           % equally spaced angles from 0 to pi
  L=line_pixel_length(lines_d(i),lines_theta(i),n_pixels);
           % L is a matrix of the same size as the image
           % with entries giving the length of the line over each pixel
  l=L(:);
           % make matrix L into a vector, as for X
  y(i)=l'*x+normrnd(0,sigma);
           % l'*x gives "line integral" of line over image,
           % that is, the intensity of each pixel is multiplied by the
           % length of line over that pixel, and then add for all pixels;
           % a random, Gaussian noise, with std sigma is added to the
           % measurement
  i=i+1;
           % for next measurement line
 end
end