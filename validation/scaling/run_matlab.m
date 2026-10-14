function run_matlab(models)
%RUN_MATLAB Time MATLAB nlmefit/nlmefitsa on the scaling datasets.
%
%   Run from the repository root (after validation/scaling/make_data.py):
%
%       addpath validation/scaling
%       run_matlab                 % all models
%       run_matlab({'mmd'})        % only these (others in matlab.json stay)
%
%   Default options, as validation/scaling/run_pynlme.py. The closed-form model
%   is called for all observations at once ('Full', with each observation's
%   group index in the last column of X, see validation/matlab/benchmark.m); the
%   ODE model is solved per subject with ode45 at the common tolerance (relative
%   1e-8, absolute 1e-10). Sizes run in increasing order; once a run takes longer
%   than 60 s the larger sizes are skipped. Runs shorter than 5 s are repeated and
%   the best of 3 is kept; the first run of each method is an untimed warm-up.
%   The multi-dose model (mmd) is solved from dose to dose with ode45.
%   Writes validation/scaling/results/matlab.json.

set(0, 'DefaultFigureVisible', 'off');
warning('off', 'all');
dataDir = fullfile('validation', 'scaling', 'data');
sizes = [10 100 1000];
if ~isempty(getenv('SCALING_SIZES'))  % e.g. SCALING_SIZES=10,100
    sizes = str2double(strsplit(getenv('SCALING_SIZES'), ','));
end
limit = 60;
methods = {'LME', 'FOCE', 'SAEM'};
% SCALING_METHODS=FOCE runs only these methods (the others' results stay).
selected = ~isempty(getenv('SCALING_METHODS'));
if selected
    methods = strsplit(getenv('SCALING_METHODS'), ',');
end
if nargin < 1
    models = {'oral', 'mm', 'mmd', 'tmdd'};
end
% Results of other models already in matlab.json stay.
outDir = fullfile('validation', 'scaling', 'results');
path = fullfile(outDir, 'matlab.json');
if isfile(path)
    results = jsondecode(fileread(path));
else
    results = struct();
end
for name = models
    if ~selected || ~isfield(results, name{1})
        results.(name{1}) = struct();
    end
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
                % SAEM runs several chains on few subjects (nlmefitsa's default),
                % so its time at the smallest size does not predict the larger ones.
                if min(times) > limit && ~(strcmp(method, 'SAEM') && n == sizes(1))
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
if ~isfolder(outDir), mkdir(outDir); end
fid = fopen(path, 'w');
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
elseif strcmp(name, 'tmdd')
    c.X = d.time;
    c.model = @tmddModel;
    c.beta0 = log([0.15 0.07 8]);
    c.args = {'ParamTransform', [1 1 1], 'REParamsSelect', [1 2]};
elseif strcmp(name, 'mmd')
    c.X = d.time;
    c.model = @mmdModel;
    c.beta0 = log([0.8 12 2.5 1]);
    c.args = {'ParamTransform', [1 1 1 1], 'REParamsSelect', [2 3]};
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

function c = mmdModel(phi, t, dose)
% The mm model with the dose added to the gut at 0, 12 and 24 h: ode45 from dose
% to dose (as validation/scaling/make_data.py). Where the system is nearly
% singular (Km near 0) ode45 does not fail but takes ever smaller steps, for hours:
% after 200000 right-hand-side evaluations the model returns NaN instead, as a
% failing integrator does in SUND.
ka = phi(1); V = phi(2); vmax = phi(3); km = phi(4);
evaluations = 0;
opts = odeset('RelTol', 1e-8, 'AbsTol', 1e-10);
[ts, order] = sort(t(:));
edges = [0 12 24 inf];
state = [0; 0];
sol = zeros(size(ts));
try
    for k = 1:3
        state(1) = state(1) + dose(1);
        inside = ts >= edges(k) & ts < edges(k + 1);
        stop = min(edges(k + 1), max(ts));
        grid = unique([edges(k); ts(inside); stop]);
        [tt, y] = ode45(@rhs, grid, state, opts);
        if numel(grid) == 2  % ode45 returns all steps for a two-point span
            y = y([1 end], :); tt = tt([1 end]);
        end
        [~, where] = ismember(ts(inside), tt);
        sol(inside) = y(where, 2);
        state = y(end, :)';
        if stop >= max(ts), break; end
    end
catch err
    if ~strcmp(err.identifier, 'scaling:stiff'), rethrow(err); end
    sol(:) = NaN;
end
c = zeros(size(t(:)));
c(order) = sol;

    function ds = rhs(~, s)
        evaluations = evaluations + 1;
        if evaluations > 200000
            error('scaling:stiff', 'ode45 needs too many steps');
        end
        ds = [-ka*s(1); ka*s(1)/V - vmax*s(2)/(km + s(2))];
    end
end

function c = tmddModel(phi, t, dose)
% The stiff TMDD model (validation/models/tmdd_sund.txt), log free drug: ode15s,
% MATLAB's solver for stiff problems.
kel = phi(1); kint = phi(2); R0 = phi(3);
kon = 100; koff = 1; kdeg = 0.1;
rhs = @(~, s) [ ...
    -kel*s(1) - (kon*s(1)*(R0 + s(2)) - koff*s(3)); ...
    -kdeg*s(2) - (kon*s(1)*(R0 + s(2)) - koff*s(3)); ...
    (kon*s(1)*(R0 + s(2)) - koff*s(3)) - kint*s(3)];
[ts, order] = sort(t(:));
opts = odeset('RelTol', 1e-8, 'AbsTol', 1e-10);
[~, sol] = ode15s(rhs, [0; ts], [dose(1); 0; 0], opts);
sol = log(sol(2:end, 1));
c = zeros(size(t(:)));
c(order) = sol;
end
