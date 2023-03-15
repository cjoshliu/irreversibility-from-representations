% Compression estimate to calculate irreversibility from latent trajectories
% of 5 different CGL simulations with 3 replicates on each condition
%
% latent_filedir: a directory of latent mean files from trained VAE models
% image_filedir: a directory of CGL simulation videos
% d_direct: irreversibility rate estimated directly
% d_half_correction: the correction using half of the trajectory
%
% reference: https://journals.aps.org/pre/abstract/10.1103/PhysRevE.85.031129


close all; clear all; clc

latent_filedir = '';
image_filedir = '';
latent_filelist = dir(latent_filedir);
image_filelist = dir(image_filedir);

d_direct = zeros(5,3);
d_half_correction = zeros(5,3);
r_index = repmat(1:3,1,5);
b_index = repelem(1:5,3);
c_index = repmat(repelem(1:5,3),1);
data_count = 0;
for file_count = 3:17
    data_count = data_count + 1;
    r_count = r_index(data_count);
    b_count = b_index(data_count);
    c_count = c_index(data_count);
    
    latents = csvread([latent_filedir latent_filelist(file_count+1).name '/latent_means.csv']);
    total_datalength = length(latents);
    
    N = 500;
    fig = zeros(64,64,N);
    for i = 1:N
        fig(:,:,i) = imread([image_filedir image_filelist(file_count).name],i);
    end
    
    %% using the L2 norm
    % with ruler frame
    reference_frame = zeros(64,64);
    reference_frame(:,:) = imread([image_filedir 'ruler.tif']);
    
    mean_pattern_distance = mean( sqrt(sum(sum((fig - reference_frame).^2))) );
    mean_latent_distance = mean( sqrt(sum((latents(1:N,:) - latents(end,:)).^2,2)) );
    latents_rescale = mean_pattern_distance/mean_latent_distance/1000 .* latents;

    range_count = 0;
    for range = [10000]
        range_count = range_count + 1;
        traj = floor(latents_rescale(1:range,:)/1);
        data_length = length(traj);
        
        compression_whole = compress_label(traj);
        cross_parsing_whole = cross_parsing_label(traj,flip(traj));
        d_direct(c_count,r_count) = 1/(data_length) * ( length(cross_parsing_whole)*log(data_length) - length(compression_whole)*log(length(compression_whole)) );
        
        compression_half = compress_label( traj( 1:round(data_length/2),: ) );
        cross_parsing_half = cross_parsing_label( traj( round(data_length/2)+1:end,: ),traj( 1:round(data_length/2)-1,: ) ); % Note that for increment, traj_1 and traj_2 may have different length
        d_half_correction(c_count,r_count) = 1/( round(data_length/2) ) * ( length(cross_parsing_half)*log( round(data_length/2) ) - length(compression_half)*log(length(compression_half)) );
        
    end
    
    
end
