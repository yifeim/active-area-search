function [] = startup()
	wd = pwd;
	cd('./ext/gpml2');
	startup;
	cd(wd)

	addpath(genpath([pwd, '/ext/gpml_extensions']));
	addpath(genpath([pwd, '/util']));
	addpath(genpath([pwd, '/src']));
end

