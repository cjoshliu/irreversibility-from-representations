% This function generates labels for the LZ77 compression algorithm

function compression = compress_label(traj_whole)

data_length = length(traj_whole);
compression = 1;
shift = 2;
count = 1;
while shift < data_length
    count = count + 1;
    
    window_size = shift - 1;
    temp = find(sum(abs(traj_whole(1:window_size,:) - traj_whole(shift,:)),2) == 0);
    index = temp;
    
    copy_length = 1;
    while ~isempty(temp)
        if index(end) == window_size
            index(end) = [];
        end
        shift = shift + 1;
        if shift > data_length
            break;
        end
        
        copy_length = copy_length + 1;
%         temp = find(traj_whole(shift,:) == traj_whole(index + 1,:));
        temp = find(sum(abs(traj_whole(index + 1,:) - traj_whole(shift,:)),2) == 0);
        index = index(temp) + 1;
        
    end
    
    shift = shift + 1;
    
    compression(count) = copy_length;
    
end

end