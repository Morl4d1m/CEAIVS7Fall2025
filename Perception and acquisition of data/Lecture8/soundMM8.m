% compare_speech_envelope_and_filtering_complete.m
% Full end-to-end: recordings or WAV load, envelope + modulation analysis,
% filtering (4k/2k/1k), spectrograms, per-signal figures + combined figure.
% Good sources for understanding:
% https://support.ircam.fr/docs/AudioSculpt/3.0/co/Spectral%20Intro_1.html
% https://www.bksv.com/media/doc/BO0187.pdf
% https://en.wikipedia.org/wiki/Envelope_(waves)


%% ---------------- Parameters ----------------
fs_target = 44100;        % working sampling rate (Hz)
rec_duration = 30;        % seconds for recording (if chosen)
plot_mod_freq_max = 50;   % show envelope spectrum up to this freq (Hz)
envelope_smooth_fc = 30;  % Hz - lowpass used to smooth analytic envelope
mod_fbands = [0 4; 4 12; 12 30]; % modulation frequency bands (Hz)

filter_cutoffs = [4000, 2000, 1000]; % low-pass cutoffs for audio filtering (Hz)
butter_order = 6;

%% ---------------- Input (record or load) ----------------
do_record = menu('Choose input method','Record microphone (sequential)','Load WAV files')==1;

sig = cell(1,2);
if do_record
    fprintf('Recording two speech samples of %d seconds each. Speak clearly.\n', rec_duration);
    pause(1);
    for k = 1:2
        fprintf('Get ready: Recording sample %d of 2 in 3 seconds...\n', k);
        pause(3);
        fprintf('SPEAK!\n')
        recObj = audiorecorder(fs_target,16,1);
        recordblocking(recObj, rec_duration);
        sig{k} = getaudiodata(recObj);
        fprintf('Recorded sample %d: %d samples at %d Hz.\n', k, length(sig{k}), recObj.SampleRate);
        fprintf('Done recording\n')
        pause(1);
    end
else
    fprintf('Select first WAV file (preferably >=30 s) — speaker 1.\n');
    [f1,p1] = uigetfile({'*.wav;*.flac;*.mp3','Audio files (*.wav,*.flac,*.mp3)';'*.*','All files'},'Select first audio file');
    if isequal(f1,0), error('No file selected. Restart the script and choose files.'); end
    [x1,fs1] = audioread(fullfile(p1,f1));
    fprintf('Loaded %s (%.1f s, %d Hz)\n', f1, length(x1)/fs1, fs1);

    fprintf('Select second WAV file — speaker 2.\n');
    [f2,p2] = uigetfile({'*.wav;*.flac;*.mp3','Audio files (*.wav,*.flac,*.mp3)';'*.*','All files'},'Select second audio file');
    if isequal(f2,0), error('No file selected. Restart the script and choose files.'); end
    [x2,fs2] = audioread(fullfile(p2,f2));
    fprintf('Loaded %s (%.1f s, %d Hz)\n', f2, length(x2)/fs2, fs2);

    % Mono conversion and resample
    if size(x1,2)>1, x1 = mean(x1,2); end
    if size(x2,2)>1, x2 = mean(x2,2); end
    if fs1~=fs_target, x1 = resample(x1, fs_target, fs1); end
    if fs2~=fs_target, x2 = resample(x2, fs_target, fs2); end

    sig{1} = x1(:);
    sig{2} = x2(:);
end

% Ensure column vectors and equal lengths
for k=1:2, sig{k} = sig{k}(:); end
minlen = min(length(sig{1}), length(sig{2}));
if minlen < fs_target*5
    warning('One or both signals are short (<5 s). Some metrics may be less stable.');
end
sig{1} = sig{1}(1:minlen);
sig{2} = sig{2}(1:minlen);
t = (0:minlen-1)'/fs_target;

% Peak-normalize (avoid dividing by zero)
for k=1:2
    mx = max(abs(sig{k}));
    if mx>0, sig{k} = sig{k}/mx; end
end

%% ---------------- Filtering (compute BEFORE plotting) ----------------
filtered = cell(2, length(filter_cutoffs));
for k = 1:2
    x = sig{k};
    for ci = 1:length(filter_cutoffs)
        fc = filter_cutoffs(ci);
        if fc >= fs_target/2
            y = x; % shouldn't occur
        else
            [b,a] = butter(butter_order, fc/(fs_target/2), 'low');
            y = filtfilt(b,a,x);
        end
        % scale to avoid clipping when writing audio
        if max(abs(y))>0
            y = y / max(abs(y)) * 0.99;
        end
        filtered{k,ci} = y;
        fname = sprintf('signal%d_lowpass_%dkHz.wav', k, round(fc/1000));
        audiowrite(fname, y, fs_target);
    end
end

%% ---------------- Analysis: envelopes, spectra, modulation bands ----------------
envs = cell(1,2);
mod_spec = cell(1,2);
spec_long = cell(1,2);
band_energy = zeros(2, size(mod_fbands,1));
total_energy_50 = zeros(2,1);

for k = 1:2
    x = sig{k};

    % Analytic envelope
    analytic = hilbert(x);
    env = abs(analytic);

    % Smooth envelope (lowpass)
    [bb,aa] = butter(4, envelope_smooth_fc/(fs_target/2),'low');
    env_smooth = filtfilt(bb,aa,env);
    env_smooth = env_smooth - mean(env_smooth);
    envs{k} = env_smooth;

    % Envelope (modulation) spectrum
    Nfft = 2^nextpow2(length(env_smooth));
    ENVf = fft(env_smooth, Nfft);
    freqs = (0:Nfft-1)*(fs_target/Nfft);
    mag_ENV = abs(ENVf)/length(env_smooth);
    mag_ENV = mag_ENV(1:floor(Nfft/2)+1);
    freq_axis = freqs(1:length(mag_ENV));
    mod_spec{k}.mag = mag_ENV;
    mod_spec{k}.f = freq_axis;

    % modulation-band energies
    S = mag_ENV.^2;
    for bidx = 1:size(mod_fbands,1)
        fb = mod_fbands(bidx,:);
        idx = find(freq_axis>=fb(1) & freq_axis<fb(2));
        if isempty(idx)
            band_energy(k,bidx) = 0;
        else
            band_energy(k,bidx) = sum(S(idx));
        end
    end
    idx50 = find(freq_axis<=plot_mod_freq_max);
    if isempty(idx50)
        total_energy_50(k) = sum(S); % fallback
    else
        total_energy_50(k) = sum(S(idx50));
    end

    % long-term power spectrum (in dB, normalized to max)
    Nfft2 = 2^nextpow2(length(x));
    X = fft(x,Nfft2);
    f_long = (0:Nfft2-1)*fs_target/Nfft2;
    magX = abs(X(1:Nfft2/2)).^2;
    magXdB = 10*log10(magX/max(magX) + eps); % avoid log10(0)
    spec_long{k}.f = f_long(1:Nfft2/2);
    spec_long{k}.dB = magXdB;
end

% modulation band percentages (row-wise)
band_pct = 100 * (band_energy ./ total_energy_50(:));

%% ---------------- Plot: per-signal figures (waveform+envelope, mod spec, long-term spec, spectrograms for original+filtered) ----------------
% Spectrogram parameters
win = 1024;
noverlap = round(0.75*win);
nfft = 2048;

for k = 1:2
    x = sig{k};
    env_smooth = envs{k};

    %% Figure A: Time & Frequency plots (waveform+envelope, modulation spec, long-term spectra)
    figA = figure('Name', sprintf('Signal %d — Time & Frequency', k), 'Position', [100 100 1100 800]);
    tA = tiledlayout(figA, 3, 1, 'TileSpacing','compact','Padding','compact');

    % Row 1: waveform + envelope (full length)
    ax1 = nexttile(tA, 1);
    plot(ax1, t, x, 'k'); hold(ax1,'on');
    ds_factor = max(1, round(fs_target/100));      % downsample envelope for plotting (~100 Hz)
    env_ds = downsample(env_smooth, ds_factor);
    t_ds = downsample(t, ds_factor);
    % scale envelope to 90% of waveform peak for visual clarity
    plot(ax1, t_ds, env_ds/max(abs(env_ds)+eps)*0.9, 'r', 'LineWidth', 1.2);
    xlabel(ax1,'Time (s)'); ylabel(ax1,'Amplitude');
    title(ax1, sprintf('Signal %d — waveform (black) and smoothed envelope (red)', k));
    xlim(ax1, [0 t(end)]);
    legend(ax1, {'waveform','envelope'}, 'Location','northeast');

    % Row 2: envelope (modulation) spectrum
    ax2 = nexttile(tA, 2);
    plot(ax2, mod_spec{k}.f, mod_spec{k}.mag, 'b', 'LineWidth', 1);
    xlim(ax2, [0 plot_mod_freq_max]); grid(ax2,'on');
    xlabel(ax2,'Modulation freq (Hz)'); ylabel(ax2,'Magnitude');
    title(ax2, 'Envelope (modulation) spectrum');

    % Row 3: Long-term magnitude spectrum (original + filtered)
    ax3 = nexttile(tA, 3);
    plot(ax3, spec_long{k}.f, spec_long{k}.dB, 'k', 'LineWidth', 1.2); hold(ax3,'on');
    colors = lines(length(filter_cutoffs));
    legendEntries = {'Original'};
    for ci = 1:length(filter_cutoffs)
        y = filtered{k,ci};
        NfftF = 2^nextpow2(length(y));
        Y = fft(y, NfftF);
        fY = (0:NfftF-1)*fs_target/NfftF;
        magY = abs(Y(1:NfftF/2)).^2;
        magYdB = 10*log10(magY/max(magY) + eps);
        plot(ax3, fY(1:NfftF/2), magYdB, 'Color', colors(ci,:), 'LineWidth', 1);
        legendEntries{end+1} = sprintf('LP %d Hz', filter_cutoffs(ci));
    end
    xlim(ax3, [0 8000]);
    xlabel(ax3,'Frequency (Hz)'); ylabel(ax3,'dB re. max');
    title(ax3,'Long-term magnitude spectrum (original + filtered)');
    legend(ax3, legendEntries, 'Location','northeast');
    grid(ax3,'on');

    drawnow;
    
    %% ---------------- Interactive playback buttons ----------------
    % We'll create buttons below the plot area of Figure A
    btnHeight = 25; % button height in pixels
    btnGap = 5;     % vertical gap
    
    % Positioning: start from bottom of figure
    figPos = get(figA,'Position'); % [left bottom width height]
    
    % Calculate normalized vertical positions for buttons
    nButtons = 1 + length(filter_cutoffs); % original + filtered
    for bi = 1:nButtons
        if bi == 1
            btnLabel = 'Play Original';
            yPos = 5; % pixels from bottom
            sigToPlay = sig{k};
        else
            btnLabel = sprintf('Play LP %d Hz', filter_cutoffs(bi-1));
            yPos = 5 + (bi-1)*(btnHeight + btnGap);
            sigToPlay = filtered{k,bi-1};
        end
        
        % Create pushbutton
        uicontrol('Style','pushbutton',...
            'Parent',figA,...
            'String',btnLabel,...
            'Units','pixels',...
            'Position',[10 yPos 120 btnHeight],...
            'BackgroundColor',[0.8 0.8 0.8],...
            'Callback',@(src,event)soundsc(sigToPlay, fs_target));
    end

    %% Figure B: Spectrograms (original + filtered)
    figB = figure('Name', sprintf('Signal %d — Spectrograms', k), 'Position', [220 120 1200 600]);
    % layout: one row original, second row filtered spectrograms (arranged horizontally)
    % We'll use tiledlayout with 2 rows: first row single tile, second row N tiles (nFilt).
    nFilt = length(filter_cutoffs);
    tB = tiledlayout(figB, 2, 1, 'TileSpacing','compact','Padding','compact');

    % Top: original spectrogram (full width)
    axTop = nexttile(tB, 1);
    [S,F,T,P] = spectrogram(x, win, noverlap, nfft, fs_target, 'yaxis');
    imagesc(axTop, T, F, 10*log10(abs(P)+eps));
    axis(axTop,'xy'); colormap(axTop,'jet'); colorbar(axTop);
    ylim(axTop, [0 min(8000, fs_target/2)]);
    xlabel(axTop,'Time (s)'); ylabel(axTop,'Frequency (Hz)');
    title(axTop, sprintf('Signal %d — Spectrogram (original)', k));

    % Bottom: filtered spectrograms arranged horizontally
    axBottomOuter = nexttile(tB, 2);
    posOuter = get(axBottomOuter, 'Position'); % [left bottom width height]
    delete(axBottomOuter);
    gap = 0.02;
    totalW = posOuter(3);
    singleW = (totalW - (nFilt-1)*gap) / nFilt;
    for ci = 1:nFilt
        left = posOuter(1) + (ci-1) * (singleW + gap);
        pos = [left, posOuter(2), singleW, posOuter(4)];
        axf = axes('Position', pos);
        y = filtered{k,ci};
        [Sf,Ff,Tf,Pf] = spectrogram(y, win, noverlap, nfft, fs_target, 'yaxis');
        imagesc(axf, Tf, Ff, 10*log10(abs(Pf)+eps));
        axis(axf,'xy'); colormap(axf,'jet'); colorbar(axf);
        ylim(axf, [0 min(8000, fs_target/2)]);
        title(axf, sprintf('LP %d Hz — spectrogram', filter_cutoffs(ci)));
        if ci==1
            ylabel(axf,'Frequency (Hz)');
        else
            set(axf,'YTickLabel',[]);
        end
        xlabel(axf,'Time (s)');
    end

    drawnow;
end

%% ---------------- Combined comparison figure ----------------
figc = figure('Name','Combined Comparison','Position',[200 200 1200 900]);
tcom = tiledlayout(figc, 3, 2, 'TileSpacing','compact','Padding','compact');

% Modulation spectra comparison
axC1 = nexttile(tcom,1,[1 2]);
plot(axC1, mod_spec{1}.f, mod_spec{1}.mag, 'b', 'LineWidth', 1.2); hold(axC1,'on');
plot(axC1, mod_spec{2}.f, mod_spec{2}.mag, 'r', 'LineWidth', 1.2);
xlim(axC1, [0 plot_mod_freq_max]); grid(axC1,'on');
xlabel(axC1,'Modulation freq (Hz)'); ylabel(axC1,'Mag');
title(axC1,'Envelope modulation spectra: Signal1 (blue) vs Signal2 (red)');
legend(axC1, {'Signal1','Signal2'});

% Long-term spectra comparison (original)
axC2 = nexttile(tcom,3,[1 2]);
plot(axC2, spec_long{1}.f, spec_long{1}.dB, 'b', 'LineWidth', 1); hold(axC2,'on');
plot(axC2, spec_long{2}.f, spec_long{2}.dB, 'r', 'LineWidth', 1);
xlim(axC2, [0 8000]); grid(axC2,'on');
xlabel(axC2,'Frequency (Hz)'); ylabel(axC2,'dB re. max');
title(axC2,'Long-term spectra: original signals');
legend(axC2, {'Signal1','Signal2'});

% Modulation-band energy bar chart
axC3 = nexttile(tcom,5);
bar(axC3, band_pct');
xticklabels(axC3, {'0–4 Hz','4–12 Hz','12–30 Hz'});
ylabel(axC3, sprintf('%% of modulation energy (<= %d Hz)', plot_mod_freq_max));
legend(axC3, {'Signal1','Signal2'}, 'Location','northwest');
title(axC3, 'Modulation-band energy distribution');

% Also show filtered comparison spectrograms for Signal1 vs Signal2 for LP 1k (example)
% This is optional but useful — show LP1k spectrograms side-by-side
axC4 = nexttile(tcom,6);
% compute short preview spectrogram for LP 1k signals (first 10 s to keep plot readable)
previewLen = min(length(sig{1}), fs_target*10);
tvec = (0:previewLen-1)/fs_target;
S1 = filtered{1,end}; S2 = filtered{2,end}; % last cutoff = 1 kHz
S1p = S1(1:previewLen); S2p = S2(1:previewLen);
[~,F1,T1,P1] = spectrogram(S1p, win, noverlap, nfft, fs_target, 'yaxis');
[~,F2,T2,P2] = spectrogram(S2p, win, noverlap, nfft, fs_target, 'yaxis');

% create two small side-by-side axes inside the tile
posOuter = get(axC4,'Position');
delete(axC4);
wsub = posOuter(3)/2 - 0.01;
axes('Position',[posOuter(1), posOuter(2), wsub, posOuter(4)]);
imagesc(T1, F1, 10*log10(abs(P1)+eps)); axis xy; ylim([0 4000]); title('Signal1 LP1k spectrogram'); xlabel('Time (s)'); ylabel('Freq (Hz)');
axes('Position',[posOuter(1)+wsub+0.02, posOuter(2), wsub, posOuter(4)]);
imagesc(T2, F2, 10*log10(abs(P2)+eps)); axis xy; ylim([0 4000]); title('Signal2 LP1k spectrogram'); xlabel('Time (s)'); set(gca,'YTickLabel',[]);
colormap('jet');

%% ---------------- Print summary metrics ----------------
fprintf('\nModulation band energy (%% of energy up to %d Hz):\n', plot_mod_freq_max);
for bidx = 1:size(mod_fbands,1)
    fprintf(' Band %d (%.0f–%.0f Hz): Signal1 = %.1f%%, Signal2 = %.1f%%\n', ...
        bidx, mod_fbands(bidx,1), mod_fbands(bidx,2), band_pct(1,bidx), band_pct(2,bidx));
end

% Simple spectral metrics
for k=1:2
    X = fft(sig{k});
    N = length(X);
    f = (0:N-1)*fs_target/N;
    mag = abs(X(1:floor(N/2)));
    faxis = f(1:floor(N/2));
    spectral_centroid(k) = sum(faxis'.*mag)/sum(mag + eps);
    idx_high = find(faxis>2000);
    high_energy_ratio(k) = sum(mag(idx_high))./sum(mag + eps);
end
fprintf('\nSpectral centroid: Signal1 = %.0f Hz, Signal2 = %.0f Hz\n', spectral_centroid(1), spectral_centroid(2));
fprintf('High-frequency (>2 kHz) energy ratio: Signal1 = %.3f, Signal2 = %.3f\n', high_energy_ratio(1), high_energy_ratio(2));

fprintf('\nWAV files saved: signal#_lowpass_#kHz.wav for each signal and cutoff in current folder.\n');

% To interpret the modulation frequency stuff see the following:
%{
plot_mod_freq_max = 50
The modulation frequency axis of the envelope spectrum only goes up to 50 Hz in the plot.
The “envelope” varies slowly — typical syllable and prosodic rhythms are in the range 2–20 Hz.
Above ~50 Hz, there’s usually little meaningful modulation energy for speech.
So this variable limits the x-axis of the modulation (envelope) spectrum plot to 0–50 Hz.
envelope_smooth_fc = 30
The envelope (from the Hilbert transform) still contains small fluctuations.
We apply a low-pass filter at 30 Hz to keep only the slower amplitude changes — roughly the rate of syllables and stress patterns — and remove fast ripples due to pitch and fine structure.
mod_fbands = [0 4; 4 12; 12 30]
Defines three modulation frequency bands for analysis of envelope energy:
0–4 Hz - phrase or prosodic rhythm (slow variations)
4–12 Hz - syllabic rate (most intelligibility)
12–30 Hz - fast envelope changes, e.g., consonant bursts

We compute how much of the total envelope energy lies in each band — this helps compare how different speakers distribute their speech rhythm energy.
%}

