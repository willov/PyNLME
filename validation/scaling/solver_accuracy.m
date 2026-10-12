function solver_accuracy()
%SOLVER_ACCURACY MATLAB's ode45 at the scaling benchmark's settings.
%
%   Relative 1e-10, absolute 1e-12, on the problems of
%   validation/scaling/solver_accuracy.py. Run from the repository root (after
%   `solver_accuracy.py write`):
%
%       addpath validation/scaling
%       solver_accuracy
%
%   Writes validation/scaling/results/solver_values_matlab.csv.

problems = readtable(fullfile('validation', 'scaling', 'data', 'solver_problems.csv'));
times = [0.5 1 2 3 4 6 8 12];
opts = odeset('RelTol', 1e-10, 'AbsTol', 1e-12);
values = zeros(height(problems), numel(times));
for i = 1:height(problems)
    p = problems(i, :);
    rhs = @(~, s) [-p.ka*s(1); p.ka*s(1)/p.V - p.vmax*s(2)/(p.km + s(2))];
    [~, y] = ode45(rhs, [0 times], [p.dose; 0], opts);
    values(i, :) = y(2:end, 2)';
end
names = repmat({'MATLAB ode45'}, 1, numel(times));
fid = fopen(fullfile('validation', 'scaling', 'results', 'solver_values_matlab.csv'), 'w');
fprintf(fid, '%s\n', strjoin(strcat('"', names, '"'), ','));
fclose(fid);
writematrix(values, fullfile('validation', 'scaling', 'results', 'solver_values_matlab.csv'), ...
    'WriteMode', 'append');
end
