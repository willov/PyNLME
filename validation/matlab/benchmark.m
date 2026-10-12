function benchmark(check)
% benchmark(true) only checks that 'Full' gives the same estimates as 'SinglePhi'.
%BENCHMARK Time MATLAB nlmefit/nlmefitsa on the benchmark cases.
%
%   Run from the repository root in MATLAB:
%
%       addpath validation/matlab
%       benchmark
%
%   The cases, start values and options are those of validation/benchmark.py
%   (tests/validation/cases.py): default options, with the model called once
%   per subject ('SinglePhi', MATLAB's default) and once for all observations
%   ('Full'). Each time is the best of several runs after a warm-up run.
%   Writes validation/reference/matlab_timings.json.
%
%   With group-level designs, 'Full' passes PHI with one row per group (and V
%   unexpanded) together with all N rows of X, so the models get each
%   observation's group index as the last column of X and expand PHI and V
%   with it (byRow).

set(0, 'DefaultFigureVisible', 'off');
dataDir = fullfile('validation', 'datasets');
cases = defineCases(dataDir);
runs = struct('lme', 10, 'foce', 3, 'saem', 3);
if nargin > 0 && check
    for i = 1:numel(cases)
        c = cases(i);
        for a = {'LME', 'FOCE'}
            b1 = nlmefit(c.X, c.y, c.group, c.V, c.model, c.beta0, c.args{:}, ...
                'ApproximationType', a{1});
            b2 = nlmefit(c.X, c.y, c.group, c.V, c.modelFull, c.beta0, c.args{:}, ...
                'ApproximationType', a{1}, 'Vectorization', 'Full');
            fprintf('%-10s %-5s max |Full - SinglePhi| = %.2g\n', c.name, a{1}, ...
                max(abs(b1 - b2)));
        end
    end
    return
end

timings = struct();
for i = 1:numel(cases)
    c = cases(i);
    t = struct();
    for vec = {'SinglePhi', 'Full'}
        if strcmp(vec{1}, 'Full')
            model = c.modelFull;
            suffix = '_full';
        else
            model = c.model;
            suffix = '_single';
        end
        fit = @(fun, extra) fun(c.X, c.y, c.group, c.V, model, c.beta0, ...
            c.args{:}, 'Vectorization', vec{1}, extra{:});
        t.(['lme' suffix]) = bestOf(runs.lme, ...
            @() fit(@nlmefit, {'ApproximationType', 'LME'}), []);
        t.(['foce' suffix]) = bestOf(runs.foce, ...
            @() fit(@nlmefit, {'ApproximationType', 'FOCE'}), []);
        if c.saem
            t.(['saem' suffix]) = bestOf(runs.saem, ...
                @() fit(@nlmefitsa, {'NChains', 1}), 1);
        end
    end
    timings.(c.name) = t;
    disp(c.name); disp(t);
end

toolbox = ver('stats');
out = struct('timings', timings, 'runs', runs, ...
    'matlab_version', version, 'toolbox_version', toolbox.Version, ...
    'threads', maxNumCompThreads, 'computer', computer);
fid = fopen(fullfile('validation', 'reference', 'matlab_timings.json'), 'w');
fprintf(fid, '%s', jsonencode(out, 'PrettyPrint', true));
fclose(fid);
end

function best = bestOf(n, fn, seed)
% Best of n timed runs, after one untimed warm-up run.
if ~isempty(seed), rng(seed, 'twister'); end
fn();
best = inf;
for k = 1:n
    if ~isempty(seed), rng(seed, 'twister'); end
    tic;
    fn();
    best = min(best, toc);
end
end

function cases = defineCases(dataDir)
cases = struct('name', {}, 'X', {}, 'y', {}, 'group', {}, 'V', {}, 'model', {}, ...
    'modelFull', {}, 'beta0', {}, 'args', {}, 'saem', {});

d = readtable(fullfile(dataDir, 'indometh.csv'));
cases(end+1) = struct('name', 'indometh', 'X', withGroup(d.time, d.subject), ...
    'y', d.conc, 'group', d.subject, 'V', [], ...
    'model', @(phi, x) biexponential(phi, x(:, 1)), ...
    'modelFull', @(phi, x) biexponential(byRow(phi, x), x(:, 1)), ...
    'beta0', [0.5 -1 2.5 0.5], 'args', {{'ParamTransform', [0 1 0 1]}}, 'saem', false);

d = readtable(fullfile(dataDir, 'theoph.csv'));
[~, first] = unique(d.subject);
cases(end+1) = struct('name', 'theoph', 'X', withGroup(d.time, d.subject), ...
    'y', d.conc, 'group', d.subject, 'V', d.dose(first), ...
    'model', @(phi, x, dose) oral(phi, x(:, 1), dose), ...
    'modelFull', @(phi, x, dose) oral(byRow(phi, x), x(:, 1), byRow(dose, x)), ...
    'beta0', [-2.5 0.5 -3.2], ...
    'args', {{'ParamTransform', [1 1 1], 'REParamsSelect', [2 3]}}, 'saem', true);

d = readtable(fullfile(dataDir, 'orange.csv'));
cases(end+1) = struct('name', 'orange', 'X', withGroup(d.age, d.tree), ...
    'y', d.circumference, 'group', d.tree, 'V', [], ...
    'model', @(phi, x) logistic(phi, x(:, 1)), ...
    'modelFull', @(phi, x) logistic(byRow(phi, x), x(:, 1)), ...
    'beta0', [192 727 348], 'args', {{'REParamsSelect', 1}}, 'saem', true);

d = readtable(fullfile(dataDir, 'linear_sim.csv'));
cases(end+1) = struct('name', 'linear', 'X', withGroup(d.x, d.id), ...
    'y', d.y, 'group', d.id, 'V', [], ...
    'model', @(phi, x) linear(phi, x(:, 1)), ...
    'modelFull', @(phi, x) linear(byRow(phi, x), x(:, 1)), ...
    'beta0', [1 1], 'args', {{'CovPattern', ones(2)}}, 'saem', true);
end

function x = withGroup(x, group)
% Append each observation's group index (in sorted group order, as nlmefit).
[~, ~, index] = unique(group);
x = [x, index];
end

function a = byRow(a, x)
% Rows of a (PHI or V) for each observation: a has one row, one row per
% observation, or one row per group (indexed by the last column of x).
if size(a, 1) ~= 1 && size(a, 1) ~= size(x, 1)
    a = a(x(:, end), :);
end
end

function f = biexponential(phi, t)
f = phi(:, 1).*exp(-phi(:, 2).*t) + phi(:, 3).*exp(-phi(:, 4).*t);
end

function f = oral(phi, t, dose)
ke = phi(:, 1); ka = phi(:, 2); cl = phi(:, 3);
f = dose.*ke.*ka./(cl.*(ka - ke)).*(exp(-ke.*t) - exp(-ka.*t));
end

function f = logistic(phi, t)
f = phi(:, 1)./(1 + exp(-(t - phi(:, 2))./phi(:, 3)));
end

function f = linear(phi, x)
f = phi(:, 1) + phi(:, 2).*x;
end
