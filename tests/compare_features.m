% Compare the C++ binaries against the k-Wave MATLAB solvers over a wider set of features than compare_with_matlab.m.
%
% Errors are relative to the maximum of each output, so outputs that are small compared with the field (a sensor
% the wave does not reach, p_final after the wave has left) show larger errors from the same round off.
%
% Each case runs the MATLAB solver, the OpenMP binary and the Metal binary, and prints the maximum error of every
% output relative to the maximum of the MATLAB output, and the difference between the two binaries. Both binaries
% should agree with MATLAB to single precision round-off (around 1e-6 to 1e-5), a GPU specific problem shows as a
% Metal error much larger than the OpenMP one.
%
% Covered: index and cuboid sensor masks, all sensor.record options (including u_non_staggered and intensity, which
% use the shifted velocity transforms), heterogeneous and homogeneous media, power law and Stokes absorption,
% nonlinearity, initial pressure, pressure sources (additive, additive-no-correction, Dirichlet, one or many signals),
% velocity sources (additive, Dirichlet) and the transducer source, in 2D and 3D, on grids with odd sizes.
%
% Run from the tests folder with: /Applications/MATLAB_R2025b.app/bin/matlab -batch compare_features

addpath('/Users/btreeby/Documents/Local-Repos/k-wave/k-Wave');
repo_dir = fileparts(fileparts(mfilename('fullpath')));
binaries = {'kspaceFirstOrder-OMP', 'kspaceFirstOrder-Metal'};
common   = {'PlotSim', false, 'DataCast', 'single', 'PMLInside', false, 'PMLSize', 10};
results  = {};

%% Case 1: 3D index sensor, all outputs, heterogeneous absorbing nonlinear medium, odd grid sizes
clear medium source sensor;
Nx = 47; Ny = 42; Nz = 39; dx = 0.1e-3;
kgrid = kWaveGrid(Nx, dx, Ny, dx, Nz, dx);
medium.sound_speed = 1500 * ones(Nx, Ny, Nz); medium.sound_speed(1:20, :, :) = 1650;
medium.density = 1000 * ones(Nx, Ny, Nz);     medium.density(:, 1:15, :) = 1150;
medium.BonA = 5 * ones(Nx, Ny, Nz);           medium.BonA(:, :, 1:10) = 8;
medium.alpha_coeff = 0.5 * ones(Nx, Ny, Nz);  medium.alpha_coeff(25:end, :, :) = 1.2;
medium.alpha_power = 1.3;
kgrid.makeTime(medium.sound_speed, 0.3, 3e-6);
source.p0 = 5 * makeBall(Nx, Ny, Nz, 22, 20, 18, 4);
rng(1);
sensor.mask = zeros(Nx, Ny, Nz); sensor.mask(randperm(Nx * Ny * Nz, 150)) = 1;
sensor.record = {'p', 'p_max', 'p_min', 'p_rms', 'p_max_all', 'p_min_all', 'p_final', ...
                 'u', 'u_max', 'u_min', 'u_rms', 'u_max_all', 'u_min_all', 'u_final', 'u_non_staggered', 'I', 'I_avg'};
results = runCase(results, '3D index all outputs', kgrid, medium, source, sensor, common, repo_dir, binaries);

%% Case 2: 3D cuboid sensor, homogeneous nonlinear lossless medium, additive velocity source
clear medium source sensor;
Nx = 40; Ny = 36; Nz = 32;
kgrid = kWaveGrid(Nx, dx, Ny, dx, Nz, dx);
medium.sound_speed = 1500; medium.density = 1000; medium.BonA = 7;
kgrid.makeTime(medium.sound_speed, 0.3, 3e-6);
source.u_mask = zeros(Nx, Ny, Nz); source.u_mask(5, 10:25, 8:20) = 1;
source.ux = 0.5 * toneBurst(1/kgrid.dt, 1.5e6, 3);
sensor.mask = [10, 5, 5, 20, 30, 12; 30, 8, 20, 35, 30, 28].';
sensor.record = {'p', 'p_max', 'p_rms', 'u', 'u_non_staggered', 'u_max'};
results = runCase(results, '3D cuboids, velocity source', kgrid, medium, source, sensor, common, repo_dir, binaries);

%% Case 3: 2D Stokes absorption, Dirichlet velocity source with many signals
clear medium source sensor;
Nx = 90; Ny = 77;
kgrid = kWaveGrid(Nx, dx, Ny, dx);
medium.sound_speed = 1500 * ones(Nx, Ny); medium.sound_speed(50:end, :) = 1600;
medium.density = 1000;
medium.alpha_coeff = 0.75; medium.alpha_power = 2; medium.alpha_mode = 'stokes';
kgrid.makeTime(medium.sound_speed, 0.3, 4e-6);
source.u_mask = zeros(Nx, Ny); source.u_mask(10, 20:40) = 1;
signal = toneBurst(1/kgrid.dt, 1e6, 3);
source.uy = 0.2 * (1 + (1:21).' / 21) * signal;
source.u_mode = 'dirichlet';
sensor.mask = makeCircle(Nx, Ny, 45, 38, 25);
sensor.record = {'p', 'u', 'u_non_staggered', 'p_final', 'u_final'};
results = runCase(results, '2D Stokes, Dirichlet velocity', kgrid, medium, source, sensor, common, repo_dir, binaries);

%% Case 4: 3D linear lossless homogeneous medium, pressure source with many signals, additive-no-correction
clear medium source sensor;
Nx = 36; Ny = 36; Nz = 36;
kgrid = kWaveGrid(Nx, dx, Ny, dx, Nz, dx);
medium.sound_speed = 1500; medium.density = 1000;
kgrid.makeTime(medium.sound_speed, 0.3, 3e-6);
source.p_mask = zeros(Nx, Ny, Nz); source.p_mask(8, 10:20, 12:16) = 1;
signal = toneBurst(1/kgrid.dt, 1e6, 3);
source.p = (1:sum(source.p_mask(:))).' / 10 * signal;
source.p_mode = 'additive-no-correction';
sensor.mask = zeros(Nx, Ny, Nz); sensor.mask(28, :, :) = 1;
sensor.record = {'p', 'p_final', 'u_non_staggered'};
results = runCase(results, '3D linear, p additive-no-corr', kgrid, medium, source, sensor, common, repo_dir, binaries);

%% Case 5: 3D Dirichlet pressure source with one signal, heterogeneous density, linear power law absorption
clear medium source sensor;
Nx = 34; Ny = 30; Nz = 28;
kgrid = kWaveGrid(Nx, dx, Ny, dx, Nz, dx);
medium.sound_speed = 1500; medium.density = 1000 * ones(Nx, Ny, Nz); medium.density(20:end, :, :) = 1300;
medium.alpha_coeff = 0.6; medium.alpha_power = 1.7;
kgrid.makeTime(medium.sound_speed, 0.3, 3e-6);
source.p_mask = zeros(Nx, Ny, Nz); source.p_mask(6, 10:20, 10:18) = 1;
source.p = toneBurst(1/kgrid.dt, 1e6, 3);
source.p_mode = 'dirichlet';
sensor.mask = zeros(Nx, Ny, Nz); sensor.mask(26, :, :) = 1;
sensor.record = {'p', 'p_rms', 'I_avg'};
results = runCase(results, '3D Dirichlet p, linear absorbing', kgrid, medium, source, sensor, common, repo_dir, binaries);

%% Case 6: 2D additive pressure source, heterogeneous sound speed, nonlinear power law absorption
clear medium source sensor;
Nx = 100; Ny = 84;
kgrid = kWaveGrid(Nx, dx, Ny, dx);
medium.sound_speed = 1500 * ones(Nx, Ny); medium.sound_speed(:, 40:end) = 1700;
medium.density = 1000; medium.BonA = 6; medium.alpha_coeff = 0.5; medium.alpha_power = 1.5;
kgrid.makeTime(medium.sound_speed, 0.3, 6e-6);
source.p_mask = zeros(Nx, Ny); source.p_mask(15, 30:50) = 1;
source.p = 2 * toneBurst(1/kgrid.dt, 1e6, 4);
% The sensor must be reached by the wave, otherwise it only records a tiny precursor and the relative errors
% (which are relative to the sensor data) are dominated by round off of the much larger field elsewhere
sensor.mask = zeros(Nx, Ny); sensor.mask(55, :) = 1;
sensor.record = {'p', 'p_max', 'p_min', 'I_avg', 'u_rms'};
results = runCase(results, '2D additive p, nonlinear absorbing', kgrid, medium, source, sensor, common, repo_dir, binaries);

%% Case 7: 3D transducer source
clear medium source sensor transducer;
Nx = 64; Ny = 48; Nz = 32;
kgrid = kWaveGrid(Nx, dx, Ny, dx, Nz, dx);
medium.sound_speed = 1540; medium.density = 1000; medium.alpha_coeff = 0.75; medium.alpha_power = 1.5;
kgrid.makeTime(medium.sound_speed, 0.3, 5e-6);
transducer.number_elements = 16;
transducer.element_width = 1;
transducer.element_length = 10;
transducer.element_spacing = 0;
transducer.radius = inf;
transducer.position = round([1, Ny/2 - 8, Nz/2 - 5]);
transducer.sound_speed = 1540;
transducer.focus_distance = 3e-3;
transducer.elevation_focus_distance = 3e-3;
transducer.steering_angle = 10;
% The C++ codes ignore the transmit apodization (the HDF5 file has no dataset for it, kspaceFirstOrder_saveToDisk
% drops it), so only Rectangular can be compared with MATLAB
transducer.transmit_apodization = 'Rectangular';
transducer.receive_apodization = 'Rectangular';
transducer.active_elements = ones(transducer.number_elements, 1);
transducer.input_signal = 1e-3 * toneBurst(1/kgrid.dt, 2e6, 3);
transducer = kWaveTransducer(kgrid, transducer);
sensor.mask = zeros(Nx, Ny, Nz); sensor.mask(40, :, :) = 1;
sensor.record = {'p', 'p_max'};
results = runCase(results, '3D transducer source', kgrid, medium, transducer, sensor, common, repo_dir, binaries);

%% Summary
fprintf('\n%-36s %-18s %12s %12s %14s\n', 'Case', 'Output', 'OMP error', 'Metal error', 'Metal vs OMP');
for i = 1:size(results, 1)
    flag = '';
    if (results{i, 4} > max(1e-4, 10 * results{i, 3})) || (results{i, 5} > 1e-4)
        flag = '  <-- check';
    end
    fprintf('%-36s %-18s %12.2e %12.2e %14.2e%s\n', results{i, 1}, results{i, 2}, results{i, 3}, results{i, 4}, ...
            results{i, 5}, flag);
end

%% Helper functions
function results = runCase(results, name, kgrid, medium, source, sensor, common, repo_dir, binaries)
    % Run one case through MATLAB and both binaries and add the errors of all outputs to the results.
    if kgrid.dim == 2
        solver = @kspaceFirstOrder2D;  solverC = @kspaceFirstOrder2DC;
    else
        solver = @kspaceFirstOrder3D;  solverC = @kspaceFirstOrder3DC;
    end
    ref = solver(kgrid, medium, source, sensor, common{:});
    out = cell(1, numel(binaries));
    for b = 1:numel(binaries)
        bin_path = fullfile(repo_dir, binaries{b});
        out{b} = solverC(kgrid, medium, source, sensor, common{:}, 'BinaryPath', bin_path, 'BinaryName', binaries{b});
    end
    if ~isstruct(ref)
        ref = struct('p', ref);
        out = cellfun(@(o) struct('p', o), out, 'UniformOutput', false);
    end
    fields = fieldnames(ref);
    for f = 1:numel(fields)
        row = {name, fields{f}, NaN, NaN, NaN};
        for b = 1:numel(binaries)
            if isfield(out{b}, fields{f})
                row{2 + b} = relErr(getField(out{b}, fields{f}), getField(ref, fields{f}));
            end
        end
        % Difference between the two binaries
        if isfield(out{1}, fields{f}) && isfield(out{2}, fields{f})
            row{5} = relErr(getField(out{2}, fields{f}), getField(out{1}, fields{f}));
        end
        results(end + 1, :) = row; %#ok<AGROW>
    end
end

function v = getField(s, name)
    % Values of a field as one column, over all elements of a struct array (one element per cuboid).
    v = arrayfun(@(e) reshape(e.(name), [], 1), s, 'UniformOutput', false);
    v = vertcat(v{:});
end

function e = relErr(a, b)
    % Maximum error relative to the maximum of the reference, Inf if the sizes differ or values are not finite.
    if ~isequal(size(a), size(b)) || ~all(isfinite(a(:)))
        e = Inf;
    else
        e = double(max(abs(a(:) - b(:))) / max(abs(b(:))));
    end
end
