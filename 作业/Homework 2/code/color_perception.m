% Data for problem on color perception
close all;
clear all;

% This defines 20 discrete wavelengths
wavelength = 380:20:760; %(nm)

% Define the vectors of cone responsiveness
% Cone sensitivies for wavelengths in the visible range.  This data is 
%  based on a graph on page 93 in Wandell's "Foundations of Vision" 
L_cone_log = [-.9 -1.1 -1 -.8 -.7 -.6 -.5 -.4 -.3 -.25 -.3  -.4 -.8 -1.2 -1.6 -1.9 -2.4 -2.8 -3.6 -4];

M_cone_log = [-.8 -.85 -.85 -.6 -.5 -.4 -.3 -.2 -.1 -.15 -.3  -.7 -1.2 -1.7 -2.2 -2.9 -3.5 -4.2 -4.8 -5.5];

S_cone_log = [-.4 -.3 -.2 -.1 -.3 -.6 -1.2 -1.8 -2.5 -3.2  -4.1 -5 -6  -7 -8 -9 -10 -10.5 -11 -11.5];

% Raise everything to the 10 power to get the actual responsitivies.  
% These vectors contain the coefficients l_i, m_i, and s_i described in the 
% problem statement from shortest to longest wavelength. 
L_coefficients = 10.^(L_cone_log);
M_coefficients = 10.^(M_cone_log);
S_coefficients = 10.^(S_cone_log);

% Plot the cone responsiveness. 
figure(1);
plot(wavelength,L_cone_log,'-*')
hold;
plot(wavelength,M_cone_log,'--x');
plot(wavelength,S_cone_log,'-o')
xlabel('Light Wavelength (nm)');
ylabel('Log Relative Sensitivity');
legend('L cones', 'M cones', 'S cones');
title('Approximate Human Cone Sensitivities');
grid;



% Spectral Power Distribution of phosphors from shortest to longest wavelength
B_phosphor = [30 35 45 75 90 100 90 75 45 35 30 26 23 21 20 19 18 17 16 15];
G_phosphor = [21 23 26 30 35 45 75 90 100 90 75 45 35 30 26 23 21 20 19 18];
R_phosphor = [15 16 17 18 19 21 23 26 30 35 45 75 90 100 90 75 45 35 30 26];

%Plot the phosphors
figure(2);
plot(wavelength, B_phosphor,'-*');
hold;
plot(wavelength, G_phosphor,'-o');
plot(wavelength, R_phosphor,'--x');
grid;
xlabel('Light wavelength (nm)');
ylabel('Spectral Power Distribution of Phosphors');
legend('B phosphor', 'G phosphor', 'R phosphor');


% This is the spectral power distribution of the test light from the 
% shortest wavelength to the longest. 
test_light = [ 58.2792 42.3496 51.5512 33.3951 43.2907 22.5950 57.9807 76.0365  52.9823 64.0526 20.9069  37.9818  78.3329  68.0846 46.1095 56.7829 79.4211 5.9183  60.2869  5.0269]';

%Plot the test light
figure(3);
plot(wavelength, test_light, '-*');
grid;
xlabel('Light wavelength (nm)');
ylabel('Spectral Power Distribution of Test Light');
title('Test light');

% Define approximate spectrums for sunlight and a tungsten bulb.  
% The [powers are in order of increasing wavelength
tungsten = [ 20 30 40 50 60 70  80  90 100 110 120 130 140 150 160 170 180 190 200 210];
sunlight = [40 70 100 160 240 220  180 160 180 180 160 140 140 140 140 130 120 116 110 110];

%Plot the specturms
figure(4);
plot(wavelength,tungsten,'-*');
hold
plot(wavelength,sunlight,'-o');
xlabel('Light wavelength (nm)');
ylabel('Spectral Power Distributions');
legend('Tungsten','Sunlight');
grid;

%(c)
A = [L_coefficients; M_coefficients; S_coefficients];
b = A * test_light;
B = A * [ R_phosphor; G_phosphor; B_phosphor;]';
coef = B \b;
coef