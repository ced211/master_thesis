%File used to inpaint or predict using the LPC
%To use the script, change the variable datapath to the folder where the audio are saved adn result to where the original and reconstructed audio will be saved
%The ground truth will be saved in a file named or_X.wav and the corresponding reconstruction is named rec_Y.wav where X,Y arestring and  X=Y
%A signal of 2s, suit for computing the odg will be saved in folder resdir/pecept_eval. The raw result in folder resdir/test
data_path = '../maestro_dataset/audio/'
res_dir = 'lpc_result/'
contextLength = 1024;
targetLength = 1024;
contextRatio = ceil(contextLength/targetLength);
maxLag = 1000;
prediction = true;

files = dir(fullfile(datapath, '*.wav'));

snr = 0;
count = 0;
skipped_value = 0;
for j=1 : size(files)
    audioFilePath = fullfile(files(j).folder, files(j).name)
    [audio, Fs] = audioread(audioFilePath);
    target = 16000;
    audio = resample(audio,target,Fs);

    t = linspace(0, pi/2, targetLength)';
    sqCos = cos(t).^2;

    previous_sig = audio(8000: 8000 + contextLength-1);
    target_sig = audio(8000 + contextLength : 8000 + contextLength + targetLength-1);
    next_sig = audio(8000 + contextLength + targetLength : 8000 +  2 * contextLength + targetLength-1);

    ab = arburg(previous_sig, maxLag);
    Zb = filtic(1,ab,previous_sig(end-(0:(maxLag-1))));
    forw_pred = filter(1,ab,zeros(1,targetLength),Zb)';

    if ~prediction
        next_sig = flipud(next_sig);
        af = arburg(next_sig, maxLag);
        Zf = filtic(1,af, next_sig(end-(0 :(maxLag-1))));
        backw_pred = flipud(filter(1,af,zeros(1,targetLength),Zf));
        sigout = sqCos.*forw_pred + flipud(sqCos).*backw_pred;
    else
        sigout = forw_pred;
    end
    filename1 = fullfile(res_dir, 'test', strcat('rec_', int2str(j), '.wav'))
    filename2 = fullfile(res_dir, 'test', strcat('or_' , int2str(j), '.wav'))
    filename3 = fullfile(res_dir, 'perceptual_test', strcat('rec_' , int2str(j), '.wav'))
    filename4 = fullfile(res_dir,'perceptual_test', strcat('or_' , int2str(j), '.wav'))

    audiowrite(filename1, sigout, target)
    audiowrite(filename2, target_sig, target)
    rec2s = cat(1, audio(1: 8000 + contextLength), sigout, audio(8000 + contextLength + targetLength: 32000));
    audiowrite(filename3, rec2s, target)
    audiowrite(filename4, audio(1:32000), target)
    size(sigout)
    size(target_sig)
    value = mySNR(sigout, target_sig)

    if ~isfinite(value)
        skipped_value = skipped_value + 1
        continue
    end
    snr = snr + value;
    count = count + 1;
end
snr = snr / count
skipped_value


