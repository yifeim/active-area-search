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
  
  u = aa.Ereward(pool_locs);
  [~, ind] = max_tiebreak(u, feasible_locs);

  if query_count == 0
     aa.set(pool_locs(ind, :), pool_vals(ind, :));
  else
     aa.update(pool_locs(ind, :), pool_vals(ind));
  end
  feasible_locs(ind) = false;

end

figure(2); clf; 
imagesc(.5:10, .5:10, 1-reshape(aa.set_region_rewards(), 10, 10));
hold on
y_predict = aa.predict_points(x_gnd);
[c,h] = contour(reshape(x_gnd(:,1), 50,50), reshape(x_gnd(:,2), 50,50), reshape(y_predict, 50,50), ...
linspace(clims(1), clims(2), 5));
clabel_precision(c,h,'%.2f');
scatter(aa.collected_locs(:,1), aa.collected_locs(:,2));
ax = gca;
ax.YDir = 'normal';
title('active area search');


return;
% % 
% % % ---------------------- LSE -----------------------
% % xqueried_lse = double.empty(0,hyp.dim);
% % yqueried_lse = double.empty(0,1);
% % 
% % for query_count = 0:queryLen-1
% %   
% %   
% %   %  plot region highprobs 
% %   [~, ~, feasible_rel, feasible_C] = utility_LSE( ...
% %     hyp, xqueried_lse, yqueried_lse, [], levels, [], [], ...
% %     misc_lse, []); 
% %   
% %   figure(4); clf;
% %   contourf(feasible_X1, feasible_X2, reshape(feasible_rel, size(feasible_X1)), [-2:2]);
% %   hold on; caxis([-2,2]); colorbar; 
% %   if ~isempty(xqueried_lse), scatter(xqueried_lse(:,1), xqueried_lse(:,2),100,colorlookup(yqueried_lse),'filled'); end
% %   for i=1:size(xqueried_lse,1)
% %     text(xqueried_lse(i,1), xqueried_lse(i,2),sprintf('%d',i),'fontsize',20);
% %   end
% %   drawnow;
% %   pause(.1);
% % 
% %   % --------------- ransac to pick the best point ------------------
% %   [a_best, x_best, feasible_rel, feasible_C] = utility_LSE( ...
% %     hyp, xqueried_lse, yqueried_lse, [], levels, [], [], ...
% %     misc_lse, []); 
% %   
% %   misc_lse.feasible_rel = feasible_rel;
% %   misc_lse.feasible_C   = feasible_C;
% %   
% %   disp([x_best]);
% %   
% %   xqueried_lse = [xqueried_lse; x_best];
% %   yqueried_lse = y_view(knnsearch(x_view, xqueried_lse), :);
% %   
% % end
% % 
% % % ---------------------- Unc -----------------------
% % xqueried_unc = double.empty(0,hyp.dim);
% % yqueried_unc = double.empty(0,1);
% % 
% % for query_count = 0:queryLen - 1
% %   
% %   % plot current state
% %   figure(5); clf;
% %   plot_post_mk(hyp,regions,levels,highprob,X1_view,X2_view,xqueried_unc, yqueried_unc);
% %   
% %   [m, k_D] = mean_gp(feasible_x, xqueried_unc, yqueried_unc, hyp);
% %   
% %   [~, ind] = max(k_D);
% %   xqueried_unc = [xqueried_unc; feasible_x(ind, :)];
% %   yqueried_unc = y_view(knnsearch(x_view, xqueried_unc), :);
% % end
% % 
% % % ----------------------- random -------------------
% % xqueried_rand = double.empty(0,hyp.dim);
% % yqueried_rand = double.empty(0,1);
% % 
% % for query_count = 0:queryLen - 1
% %   
% %   plot current state
% %   figure(6); clf;
% %   plot_post_mk(hyp,regions,levels,highprob,X1_view,X2_view,xqueried_rand, yqueried_rand);
% %   
% %   ind = randsample(feasible_n,1);
% %   xqueried_rand = [xqueried_rand; feasible_x(ind, :)];
% %   yqueried_rand = y_view(knnsearch(x_view, xqueried_rand), :);
% % end
% % 
% % % --------------------- evaluation ------------------
% % % num_regions found
% % fprintf('total nb interesting regions = %d\n', sum(view_pts_probs > highprob));
% % 
% % [R_aa,P_aa,F_aa] = precision_recall(hyp, regions, levels, highprob, view_pts_probs, ...
% %   xqueried_aa, yqueried_aa);
% % 
% % [R_lse,P_lse,F_lse] = precision_recall(hyp, regions, levels, highprob, view_pts_probs, ...
% %   xqueried_lse, yqueried_lse);
% % 
% % [R_unc,P_unc,F_unc] = precision_recall(hyp, regions, levels, highprob, view_pts_probs, ...
% %   xqueried_unc, yqueried_unc);
% % 
% % [R_rand,P_rand,F_rand] = precision_recall(hyp, regions, levels, highprob, view_pts_probs, ...
% %   xqueried_rand, yqueried_rand);
% % 
% % disp([
% % [R_aa,P_aa,F_aa]
% % [R_lse,P_lse,F_lse]
% % [R_unc,P_unc,F_unc]
% % [R_rand,P_rand,F_rand]
% % ]);
% % 
% % 
% % % num_positive area found 
% % fprintf('total nb interesting points = %d\n', sum(y_view > levels));
% % [R_pts_aa,P_pts_aa,F_pts_aa] = precision_recall_pts(...
% %   hyp, levels, highprob, xqueried_aa, yqueried_aa, x_view, y_view);
% % 
% % [R_pts_lse,P_pts_lse,F_pts_lse] = precision_recall_pts(...
% %   hyp, levels, highprob, xqueried_lse, yqueried_lse, x_view, y_view);
% % 
% % [R_pts_unc,P_pts_unc,F_pts_unc] = precision_recall_pts(...
% %   hyp, levels, highprob, xqueried_unc, yqueried_unc, x_view, y_view);
% % 
% % [R_pts_rand,P_pts_rand,F_pts_rand] = precision_recall_pts(...
% %   hyp, levels, highprob, xqueried_rand, yqueried_rand, x_view, y_view);
% % 
% % disp([
% % [R_pts_aa,P_pts_aa,F_pts_aa]
% % [R_pts_lse,P_pts_lse,F_pts_lse]
% % [R_pts_unc,P_pts_unc,F_pts_unc]
% % [R_pts_rand,P_pts_rand,F_pts_rand]
% % ]);
% % 
% % % ----------------- plot ------------------
% % figure(10); clf;
% % plot_ground_truth(hyp,regions,levels,highprob,X1_view,X2_view, y_view);
% % 
% % figure(11); clf;
% % plot_sample(hyp,regions,levels,highprob,X1_view,X2_view,xqueried_aa, yqueried_aa, {'sk','fill'});
% % 
% % figure(12); clf;
% % plot_sample(hyp,regions,levels,highprob,X1_view,X2_view,xqueried_lse, yqueried_lse, {'sk','fill'});
% % 
% % figure(13); clf;
% % plot_sample(hyp,regions,levels,highprob,X1_view,X2_view,xqueried_unc, yqueried_unc, {'sk','fill'});
% % 
% % figure(14); clf;
% % plot_sample(hyp,regions,levels,highprob,X1_view,X2_view,xqueried_rand, yqueried_rand, {'sk','fill'});
% % 
% % folder_name = 'run_compare';
% % fname_result = @(len, mtd) [folder_name, sprintf('/midnight%02d-%d.eps', len, mtd)]; 
% % for d=10:14
% %   saveas(d, fname_result(queryLen, d-10), 'psc2');
% % end
% % 
% % 
% % save([folder_name, '/', datestr(now, 30), '.mat']);
% % savecolorbar(hyp, levels, [folder_name, '/thiscolorbar.eps']);



% ----------------------- random -------------------
hyp.sf2  = hyp.gamma2 / (sqrt(2*pi)*hyp.ell)^hyp.dim; 

xqueried_mig = double.empty(0,hyp.dim);
yqueried_mig = double.empty(0,1);

for query_count = 0:queryLen - 1
  
  plot current state
  figure(7); clf;
  plot_post_mk(hyp,regions,levels,highprob,X1_view,X2_view,xqueried_mig, yqueried_mig);

  
%   ind = randsample(feasible_n,1);
%   xqueried_mig = [xqueried_mig; feasible_x(ind, :)];
%   yqueried_mig = y_view(knnsearch(x_view, xqueried_mig), :);
end
