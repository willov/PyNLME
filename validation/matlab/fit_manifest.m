function fit_manifest(manifestFile, resultsFile, approx, display)
% Fit simulated datasets listed in a manifest with MATLAB nlmefit (default
% options), for the randomized comparison in
% tests/validation/test_random_vs_matlab.py. The manifest is written by
% tests/validation/simulate.py (write_for_r); the model families mirror it.
%
%   matlab -batch "addpath validation/matlab; fit_manifest('m.json', 'r.json', 'LME')"

if nargin < 3, approx = 'LME'; end
if nargin < 4, display = 'off'; end  % 'iter' prints nlmefit's trace
cases = jsondecode(fileread(manifestFile));
results = struct();
for k = 1:numel(cases)
    c = cases(k);
    d = readtable(c.file);
    X = d.x; y = d.y; group = d.id;
    start = c.start(:)';
    switch c.family
        case 'oral'
            V = accumarray(group, d.v, [], @(v) v(1));
            model = @(phi, t, dose) dose .* phi(1) .* phi(2) ./ (phi(3) .* (phi(2) - phi(1))) ...
                .* (exp(-phi(1) .* t) - exp(-phi(2) .* t));
            args = {'ParamTransform', [1 1 1], 'REParamsSelect', [2 3]};
        case 'logistic'
            V = [];
            model = @(phi, t) phi(1) ./ (1 + exp(-(t - phi(2)) ./ phi(3)));
            args = {'REParamsSelect', [1 2]};
        case 'linear'
            V = [];
            model = @(phi, x) phi(1) + phi(2) .* x;
            args = {'CovPattern', ones(2)};
        case 'emax'
            V = [];
            model = @(phi, x) phi(1) + phi(2) .* x ./ (phi(3) + x);
            args = {'ParamTransform', [0 0 1], 'REParamsSelect', [1 3]};
    end
    lastwarn('');
    try
        [beta, psi, stats] = nlmefit(X, y, group, V, model, start, args{:}, ...
                                     'ApproximationType', approx, ...
                                     'Options', statset('Display', display));
        [msg, id] = lastwarn();
        results.(c.name) = struct('beta', beta(:)', 'psi', psi, 'logl', stats.logl, ...
            'errorparam', stats.errorparam, 'warning_id', id, 'warning', msg);
    catch err
        results.(c.name) = struct('error', err.message);
    end
end
fid = fopen(resultsFile, 'w');
fprintf(fid, '%s', jsonencode(results));
fclose(fid);
end
