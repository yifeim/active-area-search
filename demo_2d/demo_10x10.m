% true model
load demo_10x10_gnd.mat

x_shape  = [50,50];
side     = 1;
beta_t   = 9;
eps_band = .1;

gnd = ActiveAreaSearch(gp_model, gp_para, regions, level, side, highprob);
found = gnd.update(x_gnd, y_gnd);

figure(1); clf;
[~, ~, f, fs2] = gnd.predict_points(x_gnd);
plot_demo(x_shape, x_gnd, f, fs2, [], regions, level, gnd.alpha, gnd.beta2, found, 'plotTailGaussian', false);
title('ground truth');


queryLen = 25;

% ----------------------- aas ----------------------------
aas = ActiveAreaSearch(gp_model, gp_para, regions, level, side, highprob);

for query_count = 0:queryLen-1
  u = aas.utility(x_gnd);
  [~, ind] = max_tiebreak(u);

  found = aas.update(x_gnd(ind, :), y_gnd(ind, :));

end

figure(2); clf;
[~, ~, f, fs2] = aas.predict_points(x_gnd);
plot_demo(x_shape, x_gnd, f, fs2, aas.collected_locs, regions, level, aas.alpha, aas.beta2, found);
title('active area search');

% ----------------------- lse ----------------------------
lse = ActiveLevelSetEstimation(gp_model, gp_para, level, beta_t, eps_band, x_gnd);

for query_count = 0:queryLen-1
  u = lse.utility();
  [~, ind] = max_tiebreak(u);

  lse.update(x_gnd(ind, :), y_gnd(ind, :));  
end

region_measure_lse = ActiveAreaSearch(gp_model, gp_para, regions, level, side, highprob);
found = region_measure_lse.update(lse.collected_locs, lse.collected_vals);

figure(3); clf; 
[~, ~, f, fs2] = region_measure_lse.predict_points(x_gnd);
plot_demo(x_shape, x_gnd, f, fs2, lse.collected_locs, regions, level, region_measure_lse.alpha, region_measure_lse.beta2, found);
title('active level set estimation');


% ----------------------- unc ----------------------------
unc = UncertaintySampling(gp_model, gp_para);

for query_count = 0:queryLen-1
  u = unc.utility(x_gnd);
  [~, ind] = max_tiebreak(u);

  unc.update(x_gnd(ind, :), y_gnd(ind, :));  
end

region_measure_unc = ActiveAreaSearch(gp_model, gp_para, regions, level, side, highprob);
found = region_measure_unc.update(unc.collected_locs, unc.collected_vals);

figure(4); clf; 
[~, ~, f, fs2] = region_measure_unc.predict_points(x_gnd);
plot_demo(x_shape, x_gnd, f, fs2, unc.collected_locs, regions, level, region_measure_unc.alpha, region_measure_unc.beta2, found);
title('uncertainty sampling');



% ----------------------- rnd ----------------------------
rnd = RandomSampling(gp_model, gp_para);

for query_count = 0:queryLen-1
  u = rnd.utility(x_gnd);
  [~, ind] = max_tiebreak(u);

  rnd.update(x_gnd(ind, :), y_gnd(ind, :));  
end

region_measure_rnd = ActiveAreaSearch(gp_model, gp_para, regions, level, side, highprob);
found = region_measure_rnd.update(rnd.collected_locs, rnd.collected_vals);

figure(5); clf; 
[~, ~, f, fs2] = region_measure_rnd.predict_points(x_gnd);
plot_demo(x_shape, x_gnd, f, fs2, rnd.collected_locs, regions, level, region_measure_rnd.alpha, region_measure_rnd.beta2, found);
title('random sampling');
