function run_matlab()
%RUN_MATLAB Time MATLAB nlmefit/nlmefitsa on the scaling datasets.
%
%   Run from the repository root (after validation/scaling/make_data.py):
%
%       addpath validation/scaling
%       run_matlab
%
%   Default options, as validation/scaling/run_pynlme.py. The closed-form model
%   is called for all observations at once ('Full', with each observation's
%   group index in the last column of X, see validation/matlab/benchmark.m); the
%   ODE model is solved per subject with ode45 at the common tolerance (relative
%   1e-8, absolute 1e-10). Sizes run in increasing order; once a run takes longer
%   than 60 s the larger sizes are skipped. Runs shorter than 5 s are repeated and
%   the best of 3 is kept; the first run of each method is an untimed warm-up.
%   Writes validation/scaling/results/matlab.json.

set(0, 'DefaultFigureVisible', 'off');
warning('off', 'all');
dataDir = fullfile('validation', 'scaling', 'data');
sizes = [10 100 1000];
limit = 60;
methods = {'LME', 'FOCE', 'SAEM'};
results = struct();
for name = {'oral', 'mm'}
    tooSlow = {};
    for n = sizes
        c = loadCase(dataDir, name{1}, n);
        for k = 1:numel(methods)
            method = methods{k};
            key = sprintf('n%d', n);
            if ismember(method, tooSlow)
                results.(name{1}).(method).(key) = struct('skipped', true);
                continue
            end
            run = @() fitOnce(c, method);
            try
                if n == sizes(1)
                    run();  % warm-up
                end
                times = [];
                for rep = 1:3
                    tic;
                    logl = run();
                    times(end+1) = toc; %#ok<AGROW>
                    if times(end) > 5
                        break
                    end
                end
                results.(name{1}).(method).(key) = struct('time_s', min(times), 'logl', logl);
                fprintf('%s %5d %-5s %9.3f s\n', name{1}, n, method, min(times));
                if min(times) > limit
                    tooSlow{end+1} = method; %#ok<AGROW>
                end
            catch err
                results.(name{1}).(method).(key) = struct('error', err.message);
                fprintf('%s %5d %-5s error: %s\n', name{1}, n, method, err.message);
            end
        end
    end
end
toolbox = ver('stats');
results.matlab_version = version;
results.toolbox_version = toolbox.Version;
results.threads = maxNumCompThreads;
outDir = fullfile('validation', 'scaling', 'results');
if ~isfolder(outDir), mkdir(outDir); end
fid = fopen(fullfile(outDir, 'matlab.json'), 'w');
fprintf(fid, '%s', jsonencode(results, 'PrettyPrint', true));
fclose(fid);
end

function logl = fitOnce(c, method)
switch method
    case 'SAEM'
        rng(1, 'twister');
        [~, ~, stats] = nlmefitsa(c.X, c.y, c.group, c.V, c.model, c.beta0, c.args{:});
    otherwise
        [~, ~, stats] = nlmefit(c.X, c.y, c.group, c.V, c.model, c.beta0, c.args{:}, ...
            'ApproximationType', method);
end
logl = stats.logl;
end

function c = loadCase(dataDir, name, n)
d = readtable(fullfile(dataDir, sprintf('%s_%d.csv', name, n)));
[~, first, index] = unique(d.id);
c.y = d.conc;
c.group = d.id;
c.V = d.dose(first);
if strcmp(name, 'oral')
    c.X = [d.time, index];
    c.model = @(phi, x, dose) oral(byRow(phi, x), x(:, 1), byRow(dose, x));
    c.beta0 = [-2.5 0.5 -3.2];
    c.args = {'ParamTransform', [1 1 1], 'REParamsSelect', [2 3], ...
        'Vectorization', 'Full'};
else
    c.X = d.time;
    c.model = @mmModel;
    c.beta0 = log([0.8 12 2.5 1]);
    c.args = {'ParamTransform', [1 1 1 1], 'REParamsSelect', [2 3]};
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

function c = mmModel(phi, t, dose)
% Oral dose, first-order absorption, Michaelis-Menten elimination (ODE).
ka = phi(1); V = phi(2); vmax = phi(3); km = phi(4);
rhs = @(~, s) [-ka*s(1); ka*s(1)/V - vmax*s(2)/(km + s(2))];
[ts, order] = sort(t(:));
opts = odeset('RelTol', 1e-8, 'AbsTol', 1e-10);
[~, sol] = ode45(rhs, [0; ts], [dose(1); 0], opts);
sol = sol(2:end, 2);
c = zeros(size(t(:)));
c(order) = sol;
end
