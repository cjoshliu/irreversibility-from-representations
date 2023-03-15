% This function is used to generate the cross parsing label for two trajectories

function cross_parsing = cross_parsing_label(traj_1,traj_2)
% the input data structure has to be N by m matrix and N is the length

data_length = length(traj_2);
cross_parsing = 1;
shift = 1;
count = 0;
while shift <= data_length
    count = count + 1;
    
%     temp = find(traj_1(shift) == traj_2);
    temp = find(sum(abs(traj_2 - traj_1(shift,:)),2) == 0);
    index = temp;
    
    if isempty(temp)
        shift = shift + 1;
    else
        
        copy_length = 0;
        while ~isempty(temp)
            if index(end) == data_length
                index(end) = [];
            end
            shift = shift + 1;
            copy_length = copy_length + 1;
            if shift > data_length
                break;
            end
            
%             temp = find(traj_1(shift) == traj_2(index + 1));
            temp = find(sum(abs(traj_2(index + 1,:) - traj_1(shift,:)),2) == 0);
            index = index(temp) + 1;
            
        end
        cross_parsing(count) = copy_length;
        
    end
end