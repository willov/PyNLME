function run_matlab(scenarios)
%RUN_MATLAB Fit the accuracy-benchmark replicates with MATLAB nlmefit/nlmefitsa.
%
%   Run from the repository root (after validation/accuracy/simulate.py):
%
%       addpath validation/accuracy
%       run_matlab                 % both scenarios
%       run_matlab({'hard'})
%
%   Default options (with standard errors for nlmefitsa), as
%   validation/accuracy/run_pynlme.py. Writes
%   validation/accuracy/results/<scenario>/matlab.json: per method and replicate
%   the fixed effects, standard errors, random-effect variances and error
%   parameter.

if nargin < 1
    scenarios = {'basic', 'hard'};
end
set(0, 'DefaultFigureVisible', 'off');
warning('off', 'all');
methods = {'LME', 'FOCE', 'SAEM'};
for s = 1:numel(scenarios)
    scenario = scenarios{s};
    hard = strcmp(scenario, 'hard');
    dataDir = fullfile('validation', 'accuracy', 'data', scenario);
    truth = jsondecode(fileread(fullfile(dataDir, 'truth.json')));
    results = struct('LME', {{}}, 'FOCE', {{}}, 'SAEM', {{}});
    for k = 0:truth.replicates - 1
        d = readtable(fullfile(dataDir, sprintf('rep_%d.csv', k)));
        [~, first, index] = unique(d.id);
        X = [d.time, index];
        model = @(phi, x, v) oral(byRow(phi, x), x(:, 1), byRow(v(:, 1), x));
        if hard
            lwt = log(d.wt(first) / 70);
            V = [d.dose(first), lwt];
            m = numel(first);
            A = zeros(3, 4, m);
            A(1, 1, :) = 1; A(2, 2, :) = 1; A(3, 3, :) = 1;
            A(3, 4, :) = lwt;  % log(wt / 70) on log CL
            % The covariate effect starts at 0.5 in every program (see run_r.R).
            beta0 = [-2.5 0.5 -3.2 0.5];
            args = {'ParamTransform', [1 1 1], 'FEGroupDesign', A, ...
                'CovPattern', ones(3), 'ErrorModel', 'proportional', ...
                'Vectorization', 'Full'};
        else
            V = d.dose(first);
            beta0 = [-2.5 0.5 -3.2];
            args = {'ParamTransform', [1 1 1], 'REParamsSelect', [2 3], ...
                'Vectorization', 'Full'};
        end
        for j = 1:numel(methods)
            method = methods{j};
            try
                if strcmp(method, 'SAEM')
                    rng(1000 + k, 'twister');
                    [beta, psi, stats] = nlmefitsa(X, d.conc, d.id, V, model, ...
                        beta0, args{:}, 'ComputeStdErrors', true);
                else
                    [beta, psi, stats] = nlmefit(X, d.conc, d.id, V, model, ...
                        beta0, args{:}, 'ApproximationType', method);
                end
                entry = struct('beta', beta(:)', 'se', stats.sebeta(:)', ...
                    'omega2', diag(psi)', 'sigma', stats.errorparam(1));
            catch err
                entry = struct('error', err.message);
            end
            results.(method){end+1} = entry;
        end
        fprintf('%s replicate %d done\n', scenario, k);
    end
    outDir = fullfile('validation', 'accuracy', 'results', scenario);
    if ~isfolder(outDir), mkdir(outDir); end
    fid = fopen(fullfile(outDir, 'matlab.json'), 'w');
    fprintf(fid, '%s', jsonencode(results, 'PrettyPrint', true));
    fclose(fid);
end
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
