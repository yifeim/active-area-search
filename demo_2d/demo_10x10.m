% true model
load demo_10x10_gnd.mat

% minimize(gp_para, @gp, 10, gp_model.inf, gp_model.mean, gp_model.cov, gp_model.lik, x_gnd, y_gnd)

[~,~,post_gnd] = gp(gp_para, gp_model.inf, gp_model.mean, gp_model.cov, gp_model.lik, x_gnd, y_gnd);

% plot truth contour

clims = [-2,2]+level; 

figure(1); clf;
[c,h] = contour(reshape(x_gnd(:,1), 50,50), reshape(x_gnd(:,2), 50,50), reshape(y_gnd, 50,50), ...
  linspace(clims(1), clims(2), 5));
set(h,'linewidth',3);
h=clabel_precision(c,h,'%.2f'); %colorbar;
set(h,'fontsize', 20);
caxis(clims);
title('ground truth')


%debug set_region_reward
aa = ActiveAreaSearch(gp_model, gp_para, regions, level, 1, highprob);

queryLen = 25;

% ----------------------- aa ----------------------------
feasible_locs = true(size(x_gnd, 1), 1);
pool_locs = x_gnd;
pool_vals = y_gnd;
for query_count = 0:queryLen-1
  
  u = aa.utility(pool_locs);
  [~, ind] = max_tiebreak(u, feasible_locs);

  if query_count == 0
     aa.set(pool_locs(ind, :), pool_vals(ind, :));
  else
     aa.update(pool_locs(ind, :), pool_vals(ind));
  end
  feasible_locs(ind) = false;

end

figure(2); clf; 
imagesc(.5:10, .5:10, 1-reshape(aa.region_rewards(), 10, 10));
hold on
y_predict = aa.predict_points(x_gnd);
[c,h] = contour(reshape(x_gnd(:,1), 50,50), reshape(x_gnd(:,2), 50,50), reshape(y_predict, 50,50), ...
linspace(clims(1), clims(2), 5));
clabel_precision(c,h,'%.2f');
scatter(aa.collected_locs(:,1), aa.collected_locs(:,2));
ax = gca;
ax.YDir = 'normal';
title('active area search');


% ----------------------- aa ----------------------------


