% Data for beam estimation problem
m = 5;
alpha = 0.5;
det_az =[ 3  10  80  150  275];
det_el =[ 88  34  30  20  50];
p =[ 1.58  1.50  2.47  1.10  0.001];

q1 = [cosd(det_el'), cosd(det_el'), sind(det_el')];
q2 = [cosd(det_az'), sind(det_az'), ones(m, 1)];
q = q1 .* q2;

x = (alpha * q) \ p';
a = norm(x)
d = x / a;

elevation = asind(d(3))
azimuth = asind(d(2) / cosd(elevation))