function run_matlab()
%RUN_MATLAB Fit the accuracy-benchmark replicates with MATLAB nlmefit/nlmefitsa.
%
%   Run from the repository root (after validation/accuracy/simulate.py):
%
%       addpath validation/accuracy
%       run_matlab
%
%   Default options (with standard errors for nlmefitsa), as
%   validation/accuracy/run_pynlme.py. Writes
%   validation/accuracy/results/matlab.json: per method and replicate the fixed
%   effects, standard errors, random-effect variances and error SD.

set(0, 'DefaultFigureVisible', 'off');
warning('off', 'all');
dataDir = fullfile('validation', 'accuracy', 'data');
truth = jsondecode(fileread(fullfile(dataDir, 'truth.json')));
methods = {'LME', 'FOCE', 'SAEM'};
results = struct('LME', {{}}, 'FOCE', {{}}, 'SAEM', {{}});
for k = 0:truth.replicates - 1
    d = readtable(fullfile(dataDir, sprintf('rep_%d.csv', k)));
    [~, first, index] = unique(d.id);
    X = [d.time, index];
    V = d.dose(first);
    model = @(phi, x, dose) oral(byRow(phi, x), x(:, 1), byRow(dose, x));
    args = {'ParamTransform', [1 1 1], 'REParamsSelect', [2 3], 'Vectorization', 'Full'};
    for m = 1:numel(methods)
        method = methods{m};
        try
            if strcmp(method, 'SAEM')
                rng(1000 + k, 'twister');
                [beta, psi, stats] = nlmefitsa(X, d.conc, d.id, V, model, ...
                    [-2.5 0.5 -3.2], args{:}, 'ComputeStdErrors', true);
            else
                [beta, psi, stats] = nlmefit(X, d.conc, d.id, V, model, ...
                    [-2.5 0.5 -3.2], args{:}, 'ApproximationType', method);
            end
            entry = struct('beta', beta(:)', 'se', stats.sebeta(:)', ...
                'omega2', diag(psi)', 'sigma', stats.errorparam(1));
        catch err
            entry = struct('error', err.message);
        end
        results.(method){end+1} = entry;
    end
    fprintf('replicate %d done\n', k);
end
outDir = fullfile('validation', 'accuracy', 'results');
if ~isfolder(outDir), mkdir(outDir); end
fid = fopen(fullfile(outDir, 'matlab.json'), 'w');
fprintf(fid, '%s', jsonencode(results, 'PrettyPrint', true));
fclose(fid);
end

function a = byRow(a, x)
% Rows of a (PHI or V) for each observation (see validation/matlab/benchmark.m).
if size(a, 1) ~= 1 && size(a, 1) ~= size(x, 1)
    a = a(x(:, end), :);
end
end

function f = oral(phi, t, dose)
ke = phi(:, 1); ka = phi(:, 2); cl = phi(:, 3);
f = dose.*ke.*ka./(cl.*(ka - ke)).*(exp(-ke.*t) - exp(-ka.*t));
end
