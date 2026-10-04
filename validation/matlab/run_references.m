function run_references()
%RUN_REFERENCES Produce MATLAB reference fits for the pynlme validation suite.
%
%   Run from the repository root in MATLAB (Statistics and Machine Learning
%   Toolbox required):
%
%       addpath validation/matlab
%       run_references
%
%   Writes validation/reference/matlab_<case>_<method>.json for every case and
%   method, and exports the documentation datasets that ship with MATLAB
%   (orangetreedata.mat, nonlineardata.mat) to validation/datasets/.
%   The case definitions mirror tests/validation/cases.py.

% nlmefitsa draws progress plots by default; keep them off screen.
set(0, 'DefaultFigureVisible', 'off');

dataDir = fullfile('validation', 'datasets');
outDir = fullfile('validation', 'reference');
exportDocDatasets(dataDir);

cases = defineCases(dataDir);
approximations = {'LME', 'RELME', 'FO', 'FOCE'};
tight = statset('TolX', 1e-8, 'TolFun', 1e-8, 'MaxIter', 1000);

for i = 1:numel(cases)
    c = cases(i);
    for j = 1:numel(approximations)
        a = approximations{j};
        for optName = {'default', 'tight'}
            args = [c.args, {'ApproximationType', a}];
            if strcmp(optName{1}, 'tight')
                args = [args, {'Options', tight}]; %#ok<AGROW>
            end
            name = sprintf('%s_%s_%s', c.name, lower(a), optName{1});
            runOne(@nlmefit, name, c, args, a, optName{1}, [], outDir);
        end
    end
    % nlmefitsa is stochastic: record several seeds.
    for seed = 1:5
        rng(seed, 'twister');
        name = sprintf('%s_saem_seed%d', c.name, seed);
        runOne(@nlmefitsa, name, c, c.args, 'SAEM', 'default', seed, outDir);
    end
end

% Non-constant error models, on the datasets with positive responses.
errorModels = {'proportional', 'combined', 'exponential'};
for i = 1:numel(cases)
    c = cases(i);
    if ~ismember(c.name, {'theoph', 'orange'})
        continue
    end
    % Drop time 0: the oral model predicts exactly 0 there while some
    % observed concentrations are positive, which makes proportional and
    % exponential error models ill-posed. Also drop non-positive responses.
    keep = c.y > 0 & c.X(:, 1) > 0;
    c.X = c.X(keep); c.y = c.y(keep); c.group = c.group(keep);
    for e = 1:numel(errorModels)
        em = errorModels{e};
        for a = {'LME', 'RELME', 'FOCE'}
            args = [c.args, {'ApproximationType', a{1}, 'ErrorModel', em, 'Options', tight}];
            name = sprintf('%s_%s_%s_tight', c.name, lower(a{1}), em);
            runOne(@nlmefit, name, c, args, a{1}, 'tight', [], outDir);
        end
        for seed = 1:3
            rng(seed, 'twister');
            name = sprintf('%s_saem_%s_seed%d', c.name, em, seed);
            runOne(@nlmefitsa, name, c, [c.args, {'ErrorModel', em}], 'SAEM', 'default', seed, outDir);
        end
    end
end
end

function runOne(fitFun, name, c, args, approx, optName, seed, outDir)
outFile = fullfile(outDir, sprintf('matlab_%s.json', name));
if isfile(outFile)
    fprintf('%-40s exists, skipped\n', name);
    return
end
toolbox = ver('stats');
res = struct('case', c.name, 'function', func2str(fitFun), ...
    'approximation_type', approx, 'options', optName, ...
    'matlab_version', version, 'toolbox_version', toolbox.Version, ...
    'beta0', c.beta0, 'seed', seed);
try
    tic;
    [beta, psi, stats, b] = fitFun(c.X, c.y, c.group, c.V, c.model, c.beta0, args{:});
    res.elapsed_s = toc;
    res.beta = beta;
    res.psi = psi;
    res.b = b;
    res.stats = stats;
    res.error = '';
catch err
    res.error = err.message;
end
writeJson(outFile, res);
fprintf('%-40s %s\n', name, ifelse(isempty(res.error), 'ok', res.error));
end

function cases = defineCases(dataDir)
cases = struct('name', {}, 'X', {}, 'y', {}, 'group', {}, 'V', {}, ...
    'model', {}, 'beta0', {}, 'args', {});

d = readtable(fullfile(dataDir, 'indometh.csv'));
cases(end+1) = struct('name', 'indometh', 'X', d.time, 'y', d.conc, ...
    'group', d.subject, 'V', [], ...
    'model', @(phi,t) phi(1).*exp(-phi(2).*t) + phi(3).*exp(-phi(4).*t), ...
    'beta0', [1 1 1 1], 'args', {{'ParamTransform', [0 1 0 1]}});

d = readtable(fullfile(dataDir, 'theoph.csv'));
[~, first] = unique(d.subject);
pk = @(phi,t,dose) dose.*phi(1).*phi(2)./(phi(3).*(phi(2)-phi(1))) ...
    .*(exp(-phi(1).*t) - exp(-phi(2).*t));
cases(end+1) = struct('name', 'theoph', 'X', d.time, 'y', d.conc, ...
    'group', d.subject, 'V', d.dose(first), 'model', pk, ...
    'beta0', [-2.5 0.5 -3.2], ...
    'args', {{'ParamTransform', [1 1 1], 'REParamsSelect', [2 3]}});

wtc = d.wt(first) - 70;
A = repmat([1 0 0 0; 0 1 0 0; 0 0 1 0], 1, 1, numel(first));
A(3, 4, :) = wtc;
cases(end+1) = struct('name', 'theoph_wt', 'X', d.time, 'y', d.conc, ...
    'group', d.subject, 'V', d.dose(first), 'model', pk, ...
    'beta0', [-2.5 0.5 -3.2 0], ...
    'args', {{'ParamTransform', [1 1 1], 'REParamsSelect', [2 3], ...
    'FEGroupDesign', A}});

d = readtable(fullfile(dataDir, 'orange.csv'));
cases(end+1) = struct('name', 'orange', 'X', d.age, 'y', d.circumference, ...
    'group', d.tree, 'V', [], ...
    'model', @(phi,t) phi(1)./(1 + exp(-(t - phi(2))./phi(3))), ...
    'beta0', [192 727 348], 'args', {{'REParamsSelect', 1}});

d = readtable(fullfile(dataDir, 'linear_sim.csv'));
cases(end+1) = struct('name', 'linear', 'X', d.x, 'y', d.y, ...
    'group', d.id, 'V', [], 'model', @(phi,x) phi(1) + phi(2).*x, ...
    'beta0', [1 1], 'args', {{'CovPattern', ones(2)}});
end

function exportDocDatasets(dataDir)
% Datasets used by the nlmefit/nlmefitsa documentation examples. Not every
% installation ships them (they come with the documentation examples).
files = {'orangetreedata.mat', 'nonlineardata.mat'};
for k = 1:numel(files)
    if isempty(which(files{k}))
        fprintf('%s not found; skipping export\n', files{k});
        continue
    end
    s = load(files{k});
    [~, base] = fileparts(files{k});
    writeStructCsv(fullfile(dataDir, ['matlab_' base '.csv']), s);
end
end

function writeStructCsv(path, s)
% Writes every variable of s; variables with fewer rows (e.g. group-level
% predictors) are padded with NaN so they fit in one table.
names = fieldnames(s);
n = max(cellfun(@(f) size(s.(f), 1), names));
cols = {};
colNames = {};
for k = 1:numel(names)
    v = double(s.(names{k}));
    for j = 1:size(v, 2)
        col = nan(n, 1);
        col(1:size(v, 1)) = v(:, j);
        cols{end+1} = col; %#ok<AGROW>
        if size(v, 2) == 1
            colNames{end+1} = names{k}; %#ok<AGROW>
        else
            colNames{end+1} = sprintf('%s_%d', names{k}, j); %#ok<AGROW>
        end
    end
end
writetable(table(cols{:}, 'VariableNames', colNames), path);
end

function writeJson(path, value)
fid = fopen(path, 'w');
fprintf(fid, '%s', jsonencode(value, 'PrettyPrint', true));
fclose(fid);
end

function out = ifelse(cond, a, b)
if cond
    out = a;
else
    out = b;
end
end
