function check_subdirs(filepath)
% Make directories for given filepath exist,
% and try creating them if they do not.
% TODO currently just crashed if can't create dirs etc

split_path = strsplit(filepath, "/");

for i=1:(length(split_path) - 1)
	%% Go each subdirectory one by one,
	%% Checking if it exists and creating it
	%% if it does not
	cur_path = fullfile(split_path{1:i});

	if (exist(cur_path, "dir") ~= 7) 
		%% Make directory
		mkdir(cur_path);
	end
end

end
