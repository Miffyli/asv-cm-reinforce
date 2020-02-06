function success = extract_and_save(filename, input_dir, output_dir)
	addpath('CQT_toolbox_2013');
	success = 0;
	%% Hyperparams (from DEMO.m in the original ASVSpoof kit)
	B = 96;
	d = 16;
	cf = 19;
	ZsdD = 'ZsdD';

	% Extract features
	[x,fs] = audioread(strcat(input_dir, filename));

	fmax = fs/2;
	fmin = fmax/2^9;

	%% COMPUTE CQCC FEATURES
	[CQcc, LogP_absCQT, TimeVec, FreqVec, Ures_LogP_absCQT, Ures_FreqVec] = ...
	    cqcc_octave(x, fs, B, fmax, fmin, d, cf, ZsdD);

	%% Save CQcc to the output directory
	output_file = strcat(output_dir, filename);
	output_file = strrep(output_file, ".wav", ".mat");

	%% Make sure folders exist
	check_subdirs(output_file);

	%% And finally save the matrix
	save("-v6", output_file, "CQcc");
	success = 1;
end