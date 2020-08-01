% run perceptual test on a folder.
%The reference signal must be saved in a wav file names 'or_*.wav'
%For each reference signal name 'or_X.wav', the folder must contain the corresponding singal under test
%in a file named 'rec_Y.wav' where X,Y are string and  X=Y

%To use the script, change the variable audio_folder to the folder where the signals are saved and run it.
%The result will be saved in the res_dir (can be changed) file

regex = 'or*.wav'
audio_folder ='../reconstruction/pgan/0.192/percept_eval';
pqfolder = 'PQevalAudio'
files = dir(fullfile(audio_folder, regex))
target_sr = 48000
addpath(genpath(pqfolder));
ODG = zeros(2,1);
res_dir = 'odg_result.mat'

for i=1 : length(files)
    name = fullfile(files(i).folder, files(i).name);
    rec_name = strcat('rec_', files(i).name(4:end));
    rec_name = fullfile(files(i).folder, rec_name)
    [original, ~] = audioread(name);
    [reconstruction, sr] = audioread(rec_name);
    if sr ~= target_sr
        original = resample(original, target_sr, sr);
        reconstruction =  resample(reconstruction, target_sr, sr);
        audiowrite(name, original, target_sr);
        audiowrite(rec_name, reconstruction, target_sr);
    end
    odg = PQevalAudio(name, rec_name, 0, length(original))
    ODG(i) = odg
end
nanmean(ODG)
std(ODG)
save( res_dir, 'ODG')
    
    
