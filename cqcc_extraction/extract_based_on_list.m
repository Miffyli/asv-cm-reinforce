function extract_based_on_list(filelist_file, input_dir, output_dir, workers)
	% Extract CQCC features of .wav files specified in
	% filelist, and output them to the output_dir.
	% input_dir specifies root for the filelist

	addpath('CQT_toolbox_2013');

	% Read filenames
	filelist_fid = fopen(filelist_file);
	filelist = textscan(filelist_fid, "%s", "delimiter", "\n"){1};
	fclose(filelist_fid);

	fun = @(filename) extract_and_save(filename, input_dir, output_dir);

	parcellfun(workers, fun, filelist, "VerboseLevel", 2)
end
