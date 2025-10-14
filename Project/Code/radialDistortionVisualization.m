%% Radial Distortion Visualization
% This script demonstrates barrel and pincushion distortion
% and their correction using polynomial radial distortion models.

clear; close all; clc;

%% Generate base grid (undistorted image)
gridSize = 10;
[x, y] = meshgrid(linspace(-1, 1, gridSize), linspace(-1, 1, gridSize));

% Flatten grid for vectorized computation
xv = x(:);
yv = y(:);
r = sqrt(xv.^2 + yv.^2);

%% Define distortion models
% Barrel distortion (positive k1) and pincushion (negative k1)
k1_barrel = 0.3;
k1_pincushion = -0.3;

% Apply distortion model: x_d = x * (1 + k1*r^2)
xv_barrel = xv .* (1 + k1_barrel * r.^2);
yv_barrel = yv .* (1 + k1_barrel * r.^2);

xv_pincushion = xv .* (1 + k1_pincushion * r.^2);
yv_pincushion = yv .* (1 + k1_pincushion * r.^2);

%% Compute "undistorted" (corrected) coordinates
% Inverse transformation (approximation)
xv_barrel_corr = xv_barrel ./ (1 + k1_barrel * r.^2);
yv_barrel_corr = yv_barrel ./ (1 + k1_barrel * r.^2);

xv_pincushion_corr = xv_pincushion ./ (1 + k1_pincushion * r.^2);
yv_pincushion_corr = yv_pincushion ./ (1 + k1_pincushion * r.^2);

%% Plot results
figure('Color', 'w', 'Position', [200, 200, 1000, 500]);

subplot(2,2,1);
plot(xv_barrel, yv_barrel, 'b.', 'MarkerSize', 10); hold on;
plot(reshape(xv_barrel, gridSize, gridSize)', reshape(yv_barrel, gridSize, gridSize)', 'k');
plot(reshape(xv_barrel, gridSize, gridSize), reshape(yv_barrel, gridSize, gridSize), 'k');
axis equal; grid on;
title('Barrel Distortion');
xlabel('x_d'); ylabel('y_d');
xlim([-1.3 1.3]); ylim([-1.3 1.3]);

subplot(2,2,2);
plot(xv_barrel_corr, yv_barrel_corr, 'r.', 'MarkerSize', 10); hold on;
plot(reshape(xv_barrel_corr, gridSize, gridSize)', reshape(yv_barrel_corr, gridSize, gridSize)', 'k');
plot(reshape(xv_barrel_corr, gridSize, gridSize), reshape(yv_barrel_corr, gridSize, gridSize), 'k');
axis equal; grid on;
title('Corrected Barrel Distortion');
xlabel('x'); ylabel('y');
xlim([-1.3 1.3]); ylim([-1.3 1.3]);

subplot(2,2,3);
plot(xv_pincushion, yv_pincushion, 'b.', 'MarkerSize', 10); hold on;
plot(reshape(xv_pincushion, gridSize, gridSize)', reshape(yv_pincushion, gridSize, gridSize)', 'k');
plot(reshape(xv_pincushion, gridSize, gridSize), reshape(yv_pincushion, gridSize, gridSize), 'k');
axis equal; grid on;
title('Pincushion Distortion');
xlabel('x_d'); ylabel('y_d');
xlim([-1.3 1.3]); ylim([-1.3 1.3]);

subplot(2,2,4);
plot(xv_pincushion_corr, yv_pincushion_corr, 'r.', 'MarkerSize', 10); hold on;
plot(reshape(xv_pincushion_corr, gridSize, gridSize)', reshape(yv_pincushion_corr, gridSize, gridSize)', 'k');
plot(reshape(xv_pincushion_corr, gridSize, gridSize), reshape(yv_pincushion_corr, gridSize, gridSize), 'k');
axis equal; grid on;
title('Corrected Pincushion Distortion');
xlabel('x'); ylabel('y');
xlim([-1.3 1.3]); ylim([-1.3 1.3]);

sgtitle('Radial Distortion and Correction Visualization', 'FontWeight', 'bold');

