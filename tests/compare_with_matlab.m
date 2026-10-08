% Compare a kspaceFirstOrder binary against the k-Wave MATLAB solvers.
%
% Runs a 2D and a 3D simulation with heterogeneous, absorbing (power law, y = 1.5 and 1.2) and nonlinear media
% through the MATLAB solver and through the binary, then prints the maximum relative error. Single precision
% round-off gives errors of around 1e-6 to 3e-6 (this is what the macOS OpenMP build achieves).
%
% Select the binary with the environment variable KWAVE_BINARY (the OpenMP build by default). The binary is run through kspaceFirstOrder2DC / 3DC with
% the BinaryName option (kspaceFirstOrder2DG / 3DG do the same with BinaryName = kspaceFirstOrder-CUDA).
%
% Run from the tests folder with: /Applications/MATLAB_R2025b.app/bin/matlab -batch compare_with_matlab

% k-Wave must be on the MATLAB path, or set KWAVE_PATH to the k-Wave toolbox folder (the one holding kspaceFirstOrder3D.m)
if ~isempty(getenv('KWAVE_PATH'))
    addpath(getenv('KWAVE_PATH'));
end
repo_dir = fileparts(fileparts(mfilename('fullpath')));
bin_name = getenv('KWAVE_BINARY');  % e.g. KWAVE_BINARY=kspaceFirstOrder-Metal matlab -batch compare_with_matlab
if isempty(bin_name)
    bin_name = 'kspaceFirstOrder-OMP';
end
bin_path = fullfile(repo_dir, bin_name);
cpp_args = {'BinaryPath', bin_path, 'BinaryName', bin_name};
common   = {'PlotSim', false, 'DataCast', 'single', 'PMLInside', false};
err = @(a, b) max(abs(a(:) - b(:))) / max(abs(b(:)));

%% 2D: heterogeneous, absorbing, nonlinear, initial pressure
Nx = 128; Ny = 128; dx = 0.1e-3;
kgrid = kWaveGrid(Nx, dx, Ny, dx);
medium.sound_speed = 1500 * ones(Nx, Ny);
medium.sound_speed(1:Nx/2, :) = 1800;
medium.density = 1000 * ones(Nx, Ny);
medium.density(:, Ny/4:end) = 1200;
medium.alpha_coeff = 0.75;
medium.alpha_power = 1.5;
medium.BonA = 6;
kgrid.makeTime(medium.sound_speed);
source.p0 = 3 * makeDisc(Nx, Ny, Nx/2, Ny/2, 5);
sensor.mask = makeCircle(Nx, Ny, Nx/2, Ny/2, 40);
sensor.record = {'p', 'p_max', 'u'};

ref = kspaceFirstOrder2D(kgrid, medium, source, sensor, common{:});
cpp = kspaceFirstOrder2DC(kgrid, medium, source, sensor, common{:}, cpp_args{:});
fprintf('2D p     rel err: %g\n', err(cpp.p, ref.p));
fprintf('2D p_max rel err: %g\n', err(cpp.p_max, ref.p_max));
fprintf('2D ux    rel err: %g\n', err(cpp.ux, ref.ux));
fprintf('2D finite: %d\n', all(isfinite(cpp.p(:))));

%% 3D: heterogeneous, absorbing, time-varying pressure source
clear medium source sensor;
Nx = 64; Ny = 64; Nz = 64; dx = 0.1e-3;
kgrid = kWaveGrid(Nx, dx, Ny, dx, Nz, dx);
medium.sound_speed = 1500 * ones(Nx, Ny, Nz);
medium.sound_speed(1:Nx/2, :, :) = 1700;
medium.density = 1000;
medium.alpha_coeff = 0.5;
medium.alpha_power = 1.2;
kgrid.makeTime(medium.sound_speed, 0.3, 4e-6);
source.p_mask = zeros(Nx, Ny, Nz);
source.p_mask(Nx/4, Ny/2-5:Ny/2+5, Nz/2-5:Nz/2+5) = 1;
source.p = toneBurst(1/kgrid.dt, 1e6, 3);
sensor.mask = zeros(Nx, Ny, Nz);
sensor.mask(3*Nx/4, :, :) = 1;
sensor.record = {'p', 'p_rms'};

ref = kspaceFirstOrder3D(kgrid, medium, source, sensor, common{:});
cpp = kspaceFirstOrder3DC(kgrid, medium, source, sensor, common{:}, cpp_args{:});
fprintf('3D p     rel err: %g\n', err(cpp.p, ref.p));
fprintf('3D p_rms rel err: %g\n', err(cpp.p_rms, ref.p_rms));
fprintf('3D finite: %d\n', all(isfinite(cpp.p(:))));
