% DCT Basis Grid — single-image renderer (robust)
% Produces an M×M grid of 2D DCT basis blocks, each block is N×N pixels.

clear; close all; clc;

% Parameters (change these if you want)
N = 12;   % size of each basis block (pixels per tile)
M = 12;   % how many tiles per row/column (frequencies 0..M-1)
sep_px = 1; % width of separator lines between tiles (pixels). Set 0 to remove.

tile = N;
bigSize = M * tile;
big = zeros(bigSize, bigSize);

% Precompute coordinates for a single tile
[x, y] = meshgrid(0:tile-1, 0:tile-1);

% Fill the big image: rows correspond to vertical freq v, columns to horizontal freq u
for v = 0:(M-1)       % vertical frequency index (rows of tiles)
    for u = 0:(M-1)   % horizontal frequency index (cols of tiles)
        Cu = (u==0) * sqrt(1/tile) + (u~=0) * sqrt(2/tile);
        Cv = (v==0) * sqrt(1/tile) + (v~=0) * sqrt(2/tile);
        basis = Cu * Cv .* cos((pi*(2*x+1)*u) / (2*tile)) .* ...
                           cos((pi*(2*y+1)*v) / (2*tile));
        r = v*tile + (1:tile);
        c = u*tile + (1:tile);
        big(r, c) = basis;
    end
end

% Normalize globally for consistent contrast
cmax = max(abs(big(:)));
big = big / cmax;   % now in [-1, 1]

% Optional: add white separators so tile borders are visible (similar to your example)
if sep_px > 0
    for k = 1:(M-1)
        rr = k*tile + (1:sep_px);
        cc = k*tile + (1:sep_px);
        big(rr, :) = 1;   % horizontal separators (white)
        big(:, cc) = 1;   % vertical separators (white)
    end
end

% Display
figure('Color','w','Position',[100 100 700 700]);
imagesc(big, [-1 1]);            % use symmetric color limits for zero-centered contrast
colormap(gray);
axis image off;
set(gca,'Position',[0 0 1 1]);   % fill the whole figure with the image
title(sprintf('2D DCT basis — %dx%d tiles of %dx%d', M, M, tile, tile), 'FontSize', 14);

% (optional) Save to file:
% imwrite( uint8( (big + 1)/2 * 255 ), 'dct_basis_grid.png' );
