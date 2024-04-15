close all; clear all; clc

%% Change these based on which experiment you are analyzing
latent_mean_path = '/path/to/latent_means.csv';
latent_logvar_path = '/path/to/latent_logvars.csv';
image_path = '/path/to/phase_field_video.tif';
ruler_path = '/path/to/ruler.tif';
save_path = '/dir/to/save/zm/estimates/';
frame_rate = 1; % frame rate of TIFF phase field in Hz
coarse_para = 1; % coarse-graining kernel size
parsed_length = 10000; % maximum length to parse

%% Load the latent file
latents_means = readmatrix(latent_mean_path);
latents_logvars = readmatrix(latent_logvar_path);
latents = latents_means ./ sqrt(exp(latents_logvars));
total_datalength = length(latents);

%% Transform the phase field image into its complex-exponentiated form
fig = zeros(64,64,2,total_datalength-1); 
for i = 1:total_datalength-1
    phases = imread(image_path,i);
    phases = double(phases) ./ 255 .* 2 .* pi - pi;

    cosine_phase = uint8((cos(phases) + 1) ./ 2 .* 255);
    sine_phase = uint8((sin(phases) + 1) ./ 2 .* 255);
    sine_phase((phases < 0) & (sine_phase == 128)) = 127;

    fig(:,:,1,i) = double(cosine_phase)./255;
    fig(:,:,2,i) = double(sine_phase)./255;
end

reference_frame = zeros(64,64,2);
reference_frame(:,:,1) = double(imread(ruler_path,1))/255;
reference_frame(:,:,2) = double(imread(ruler_path,2))/255;

%% Use mean L2 distance from ruler to rescale latent trajectory
mean_pattern_distance = mean( sqrt(sum(sum((fig - reference_frame).^2))) ,'all');
mean_latent_distance = mean( sqrt(sum((latents(1:total_datalength-1,:) - latents(end,:)).^2,2)) );
latents_rescale = mean_pattern_distance/mean_latent_distance/5 .* latents;

%% Discretize rescaled latent trajectory
disc_traj = floor(latents_rescale(1:parsed_length,:)/coarse_para);

%% Estimate irreversibility
compression_whole = compress_label(disc_traj);
cross_parsing_whole = cross_parsing_label(disc_traj,flip(disc_traj));
d_direct = 1/(parsed_length) * ( length(cross_parsing_whole)*log(parsed_length) - length(compression_whole)*log(length(compression_whole)) ) * frame_rate;

compression_half = compress_label( disc_traj( 1:round(parsed_length/2),: ) );
cross_parsing_half = cross_parsing_label( disc_traj( round(parsed_length/2)+1:end,: ),disc_traj( 1:round(parsed_length/2)-1,: ) ); % Note that for increment, traj_1 and traj_2 may have different length
d_half_correction = 1/( round(parsed_length/2) ) * ( length(cross_parsing_half)*log( round(parsed_length/2) ) - length(compression_half)*log(length(compression_half)) ) * frame_rate;

%% Save output
save([save_path 'd_direct'],"d_direct")
save([save_path 'd_half_correction'],"d_half_correction")
