function L=line_pixel_length(d,theta,n)

% image reconstruction from line measurements (tomography via least-squares)
%
% given a grid of n by n square pixels, and a line over that grid,
% this function computes the length of line that goes over each pixel
%
% INPUTS
% d:     displacement of line,
%        distance of line from center of image,
%        measured in pixel lengths (and orthogonally to line)
% theta: angle of line,
%        measured in radians from x-axis
% n:     image size is n by n
%
% OUTPUTS
% L:     matrix of size n by n (same as image),
%        length of the line over each pixel (most entries are zero)
%
% expects an angle theta in [0,pi]
% (but will work at least for angles in [-pi/4,pi])


  % for angle in [pi/4,3*pi/4],
  % flip along diagonal (transpose) and call recursively
if theta>pi/4 & theta<3/4*pi
  L=line_pixel_length(d,pi/2-theta,n)';
  return
end

  % for angle in [3*pi/4,pi],
  % redefine line to go in opposite direction
if theta>pi/2
  d=-d;
  theta=theta-pi;
end

  % for angle in [-pi/4,0],
  % flip along x-axis (up/down) and call recursively
if theta<0
  L=flipud(line_pixel_length(-d,-theta,n));
  return
end

if theta>pi/2 | theta<0
  disp('invalid angle')
  return
end

L=zeros(n,n);

ct=cos(theta);
st=sin(theta);

x0=n/2-d*st;
y0=n/2+d*ct;

y=y0-x0*st/ct;
jy=ceil(y);
dy=rem(y+n,1);

for jx=1:n
  dynext=dy+st/ct;
  if dynext<1
    if jy>=1 & jy<=n, L(n+1-jy,jx)=1/ct; end
    dy=dynext;
  else
    if jy>=1 & jy<=n, L(n+1-jy,jx)=(1-dy)/st; end
    if jy+1>=1 & jy+1<=n, L(n+1-(jy+1),jx)=(dynext-1)/st; end
    dy=dynext-1;
    jy=jy+1;
  end
end