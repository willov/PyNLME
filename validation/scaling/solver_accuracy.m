function solver_accuracy()
%SOLVER_ACCURACY MATLAB's ODE solvers at the scaling benchmark's settings.
%
%   Relative 1e-8, absolute 1e-10, on the problems of
%   validation/scaling/solver_accuracy.py: ode45 for the non-stiff mm model,
%   ode15s for the stiff tmdd model. Run from the repository root (after
%   `solver_accuracy.py write`):
%
%       addpath validation/scaling
%       solver_accuracy
%
%   Writes validation/scaling/results/solver_values_matlab_<model>.csv.

opts = odeset('RelTol', 1e-8, 'AbsTol', 1e-10);

problems = readtable(fullfile('validation', 'scaling', 'data', 'solver_problems_mm.csv'));
times = [0.5 1 2 3 4 6 8 12];
values = zeros(height(problems), numel(times));
for i = 1:height(problems)
    p = problems(i, :);
    rhs = @(~, s) [-p.ka*s(1); p.ka*s(1)/p.V - p.vmax*s(2)/(p.km + s(2))];
    [~, y] = ode45(rhs, [0 times], [p.dose; 0], opts);
    values(i, :) = y(2:end, 2)';
end
write('mm', 'MATLAB ode45', values);

problems = readtable(fullfile('validation', 'scaling', 'data', 'solver_problems_tmdd.csv'));
times = [0.5 1 2 4 8 12 24 48 72 96];
values = zeros(height(problems), numel(times));
for i = 1:height(problems)
    p = problems(i, :);
    rhs = @(~, s) [ ...
        -p.kel*s(1) - (100*s(1)*(p.R0 + s(2)) - s(3)); ...
        -0.1*s(2) - (100*s(1)*(p.R0 + s(2)) - s(3)); ...
        (100*s(1)*(p.R0 + s(2)) - s(3)) - p.kint*s(3)];
    [~, y] = ode15s(rhs, [0 times], [p.dose; 0; 0], opts);
    values(i, :) = log(y(2:end, 1))';
end
write('tmdd', 'MATLAB ode15s', values);
end

function write(model, name, values)
path = fullfile('validation', 'scaling', 'results', sprintf('solver_values_matlab_%s.csv', model));
names = repmat({name}, 1, size(values, 2));
fid = fopen(path, 'w');
fprintf(fid, '%s\n', strjoin(strcat('"', names, '"'), ','));
fclose(fid);
writematrix(values, path, 'WriteMode', 'append');
end
