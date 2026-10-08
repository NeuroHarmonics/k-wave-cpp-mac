% Generate the benchmark input files bench_128.h5 and bench_256.h5 in the current folder.
%
% 3D heterogeneous, absorbing (y = 1.5), nonlinear medium, initial pressure source, one sensor plane, 200 time steps.
% These are the inputs used for the timings in the README (bench_256.h5 is about 400 MB).
%
% Run with: /Applications/MATLAB_R2025b.app/bin/matlab -batch make_bench_inputs

addpath('/Users/btreeby/Documents/Local-Repos/k-wave/k-Wave');
for N = [128 256]
  clear medium source sensor
  kgrid = kWaveGrid(N, 1e-4, N, 1e-4, N, 1e-4);
  medium.sound_speed = 1500 * ones(N, N, N); medium.sound_speed(1:N/2, :, :) = 1800;
  medium.density = 1000 * ones(N, N, N);     medium.density(:, 1:N/2, :) = 1200;
  medium.alpha_coeff = 0.75; medium.alpha_power = 1.5; medium.BonA = 6;
  kgrid.setTime(200, 1e-8);
  source.p0 = makeBall(N, N, N, N/2, N/2, N/2, 5);
  sensor.mask = zeros(N, N, N); sensor.mask(N/4, :, :) = 1;
  kspaceFirstOrder3D(kgrid, medium, source, sensor, 'PlotSim', false, 'DataCast', 'single', ...
      'PMLInside', true, 'SaveToDisk', sprintf('bench_%d.h5', N));
end
