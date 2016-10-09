
% true model
gp_model = struct('inf',@infExact, 'mean', @meanZero, 'cov', @covSEiso, 'lik', @likGauss);
gp_para = struct('mean', [], 'cov', [0;0], 'lik', log(.1));

[region_X1, region_X2] = meshgrid(0:9, 0:9);
regions  = [region_X1(:), region_X1(:)+1, region_X2(:), region_X2(:)+1];

[X1,X2]  = meshgrid(linspace(0,10,50), linspace(0,10,50));
x_gnd    = [X1(:), X2(:)];

% aas params
level    = 1;
side     = 1;
highprob = .8;

% lse params
beta_t   = 9;
eps_band = .1;

K        = gp_model.cov(gp_para.cov, x_gnd);
n        = size(x_gnd, 1);

num_runs = 10;
queryLen = 80;

recall_aas = nan(num_runs, queryLen);
recall_lse = nan(num_runs, queryLen);
recall_unc = nan(num_runs, queryLen);
recall_rnd = nan(num_runs, queryLen);

for run_id = 1:num_runs
   y_gnd = chol(K + exp(2*gp_para.lik) * eye(n))' * randn(n,1);

   % ground truth
   gnd = ActiveAreaSearch(gp_model, gp_para, regions, level, side, highprob);
   region_outcome_gnd = gnd.update(x_gnd, y_gnd);

   aas = ActiveAreaSearch        (gp_model, gp_para, regions, level, side, highprob);
   lse = ActiveLevelSetEstimation(gp_model, gp_para, level, beta_t, eps_band, x_gnd);
   unc = UncertaintySampling(gp_model, gp_para);
   rnd = RandomSampling(gp_model, gp_para);

   region_measure_lse = ActiveAreaSearch(gp_model, gp_para, regions,level, side, highprob);
   region_measure_unc = ActiveAreaSearch(gp_model, gp_para, regions, level, side, highprob);
   region_measure_rnd = ActiveAreaSearch(gp_model, gp_para, regions, level, side, highprob);

   for query_count = 0:queryLen-1
      % aas
      u = aas.utility(x_gnd);
      [~, ind] = max_tiebreak(u,[],false);
      aas.update(x_gnd(ind, :), y_gnd(ind, :));

      recall_aas(run_id, query_count+1) = (0+aas.cumfound>0)'*region_outcome_gnd / sum(region_outcome_gnd);

      % lse
      u = lse.utility();
      [~, ind] = max_tiebreak(u,[],false);
      lse.update(x_gnd(ind, :), y_gnd(ind, :));

      region_measure_lse.update(x_gnd(ind, :), y_gnd(ind, :));

      recall_lse(run_id, query_count+1) = (0+region_measure_lse.cumfound>0)'*region_outcome_gnd  / sum(region_outcome_gnd);

      % unc
      u = unc.utility(x_gnd);
      [~, ind] = max_tiebreak(u,[],false);
      unc.update(x_gnd(ind, :), y_gnd(ind, :));

      region_measure_unc.update(x_gnd(ind, :), y_gnd(ind, :));

      recall_unc(run_id, query_count+1) = (0+region_measure_unc.cumfound>0)'*region_outcome_gnd / sum(region_outcome_gnd);

      % rand
      u = rnd.utility(x_gnd);
      [~, ind] = max_tiebreak(u,[],false);
      rnd.update(x_gnd(ind, :), y_gnd(ind, :));

      region_measure_rnd.update(x_gnd(ind, :), y_gnd(ind, :));

      recall_rnd(run_id, query_count+1) = (0+region_measure_rnd.cumfound>0)'*region_outcome_gnd / sum(region_outcome_gnd);
   end

   [recall_aas(run_id, end), recall_lse(run_id, end), recall_unc(run_id, end), recall_rnd(run_id, end)]

end

figure;
errorbar(1:queryLen, mean(recall_aas, 1), std(recall_aas, 1)/sqrt(num_runs))
hold on
errorbar(1:queryLen, mean(recall_lse, 1), std(recall_lse, 1)/sqrt(num_runs))
errorbar(1:queryLen, mean(recall_unc, 1), std(recall_unc, 1)/sqrt(num_runs))
errorbar(1:queryLen, mean(recall_rnd, 1), std(recall_rnd, 1)/sqrt(num_runs))
legend('aas','lse','unc','rand');
grid on;
