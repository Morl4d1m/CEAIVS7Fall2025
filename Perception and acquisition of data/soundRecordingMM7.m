%% AUDIO RECORDING AND ANALYSIS EXERCISES (Enhanced)
% This script records audio signals, analyzes their time/frequency behavior,
% includes spectrograms, allows playback, and investigates dynamic range.

clear; close all; clc;

%% --- 1. Record 10-second audio ---
disp('--- Recording 1: 10 seconds with default settings ---');

fs1 = 44100;          % Sampling frequency [Hz]
nBits1 = 16;          % Bit depth
nChannels = 1;        % Mono
recObj1 = audiorecorder(fs1, nBits1, nChannels);

disp('Recording... Speak or make some sound.');
recordblocking(recObj1, 10);  % Record for 10 seconds
disp('Recording complete.');

y1 = getaudiodata(recObj1);  % Get audio data
t1 = (0:length(y1)-1)/fs1;   % Time vector

% Playback option
playChoice = input('Play back recording 1? (y/n): ', 's');
if lower(playChoice) == 'y'
    sound(y1, fs1);
end

% Plot amplitude vs time
figure('Name','Recording 1: Time Domain');
plot(t1, y1);
xlabel('Time [s]');
ylabel('Amplitude');
title(sprintf('Recording 1: Time Domain (fs = %d Hz, %d-bit)', fs1, nBits1));
grid on;

% FFT and phase
N1 = length(y1);
Y1 = fft(y1);
f1 = (0:N1-1)*(fs1/N1);
magnitude1 = abs(Y1)/max(abs(Y1));
phase1 = unwrap(angle(Y1));

% Amplitude and phase plots
figure('Name','Recording 1: Frequency Response');
subplot(2,1,1);
plot(f1(1:N1/2), magnitude1(1:N1/2));
xlabel('Frequency [Hz]');
ylabel('Normalized Amplitude');
title('Amplitude Spectrum');
xlim([0 20000])
grid on;

subplot(2,1,2);
plot(f1(1:N1/2), phase1(1:N1/2));
xlabel('Frequency [Hz]');
ylabel('Phase [radians]');
title('Phase Response');
grid on;

% 2D Spectrogram
figure('Name','Recording 1: Spectrogram');
spectrogram(y1, 1024, 512, 1024, fs1, 'yaxis');
title(sprintf('Spectrogram (fs = %d Hz, %d-bit)', fs1, nBits1));
colorbar;


%% --- 2. Second recording with lower sampling frequency and bit depth ---
disp('--- Recording 2: Different sampling frequency and bit depth ---');

fs2 = 8000;     % Lower sampling rate
nBits2 = 8;     % Lower resolution
recObj2 = audiorecorder(fs2, nBits2, nChannels);

disp('Recording again...');
recordblocking(recObj2, 10);
disp('Recording complete.');

y2 = getaudiodata(recObj2);
t2 = (0:length(y2)-1)/fs2;

% Playback option
playChoice = input('Play back recording 2? (y/n): ', 's');
if lower(playChoice) == 'y'
    sound(y2, fs2);
end

% Time-domain comparison
figure('Name','Time Domain Comparison');
subplot(2,1,1);
plot(t1, y1);
title(sprintf('Recording 1 (fs = %d Hz, %d-bit)', fs1, nBits1));
xlabel('Time [s]'); ylabel('Amplitude'); grid on;

subplot(2,1,2);
plot(t2, y2);
title(sprintf('Recording 2 (fs = %d Hz, %d-bit)', fs2, nBits2));
xlabel('Time [s]'); ylabel('Amplitude'); grid on;

% Frequency spectra comparison
N2 = length(y2);
Y2 = fft(y2);
f2 = (0:N2-1)*(fs2/N2);
magnitude2 = abs(Y2)/max(abs(Y2));

figure('Name','Frequency Spectrum Comparison');
hold on;
plot(f1(1:N1/2), magnitude1(1:N1/2), 'b', 'DisplayName','Recording 1');
plot(f2(1:N2/2), magnitude2(1:N2/2), 'r', 'DisplayName','Recording 2');
xlabel('Frequency [Hz]');
ylabel('Normalized Amplitude');
title('Comparison of Frequency Spectra');
xlim([0 20000])
legend show;
grid on;

% Spectrograms for both recordings
figure('Name','Spectrogram Comparison');
subplot(2,1,1);
spectrogram(y1, 1024, 512, 1024, fs1, 'yaxis');
title(sprintf('Spectrogram - Recording 1 (fs=%d Hz, %d-bit)', fs1, nBits1));
colorbar;

subplot(2,1,2);
spectrogram(y2, 512, 256, 512, fs2, 'yaxis');
title(sprintf('Spectrogram - Recording 2 (fs=%d Hz, %d-bit)', fs2, nBits2));
colorbar;


%% --- 3. Dynamic Range Test ---
disp('--- Dynamic Range Test ---');
disp('Record the quietest sound you can make...');
pause(1);
recordblocking(recObj1, 3);
quiet = getaudiodata(recObj1);

disp('Now record the loudest sound you can make...');
pause(1);
recordblocking(recObj1, 3);
loud = getaudiodata(recObj1);

% Optionally playback
playChoice = input('Play back quiet and loud recordings? (y/n): ', 's');
if lower(playChoice) == 'y'
    disp('Playing quiet sound...');
    sound(quiet, fs1);
    pause(4);
    disp('Playing loud sound...');
    sound(loud, fs1);
end

% RMS and dynamic range
rms_quiet = rms(quiet);
rms_loud = rms(loud);
dynamic_range_dB = 20 * log10(rms_loud / rms_quiet);

fprintf('\nRMS (quiet): %.6f\n', rms_quiet);
fprintf('RMS (loud):  %.6f\n', rms_loud);
fprintf('Estimated dynamic range: %.2f dB\n', dynamic_range_dB);

% Plot quiet vs loud signals
figure('Name','Dynamic Range Comparison');
subplot(2,1,1);
plot(quiet);
title('Quiet Sound');
xlabel('Samples'); ylabel('Amplitude'); grid on;

subplot(2,1,2);
plot(loud);
title('Loud Sound');
xlabel('Samples'); ylabel('Amplitude'); grid on;

disp('Dynamic range test complete.');
disp('The dynamic range (in dB) represents how much louder the loud sound is compared to the quiet one.');
