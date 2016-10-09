
% true model
gp_model = struct('inf',@infExact, 'mean', @meanZero, 'cov', @covSEiso, 'lik', @likGauss);
gp_para = struct('mean', [], 'cov', [log(.3);log(2)], 'lik', log(.1));
level = 1;

[X1,X2] = meshgrid(0:.02:1, 0:.02:1);
x_gnd = [X1(:), X2(:)];
n = size(x_gnd, 1);
x_shape = size(X1);

K = gp_model.cov(gp_para.cov, x_gnd);

y_gnd = chol(K + exp(2*gp_para.lik) * eye(n))' * randn(n,1);

queryLen = 20;

% ----------------------- aas ----------------------------
aas = ActiveAreaSearch(gp_model, gp_para, [0,1,0,1], level, 1, .8);

for query_count = 0:queryLen-1

   u = aas.utility(x_gnd);
   [~, ind] = max_tiebreak(u);

   aas.update(x_gnd(ind, :), y_gnd(ind, :));


   figure(2); clf; 
   y_predict = aas.predict_points(x_gnd);
   [c,h] = contour(reshape(x_gnd(:,1), x_shape), reshape(x_gnd(:,2), x_shape), reshape(y_predict, x_shape), ...
      linspace(-1, 3, 5));
   clabel(c,h);
   hold on;
   scatter(aas.collected_locs(:,1), aas.collected_locs(:,2), 50, 'k','s','filled');
   ax = gca;
   ax.YDir = 'normal';
   title('AAS choices and posterior');

   figure(3); clf; 
   imagesc(reshape(u, x_shape));
   ax = gca;
   ax.YDir = 'normal';
   title('AAS utility');

   pause(1);
end

% ----------------------- lse ----------------------------
lse = ActiveLevelSetEstimation(gp_model, gp_para, level, 9, .1, x_gnd);

for query_count = 0:queryLen-1

   u = lse.utility();
   [~, ind] = max_tiebreak(u);

   lse.update(x_gnd(ind, :), y_gnd(ind, :));


   figure(4); clf; 
   y_predict = lse.predict_points(x_gnd);
   [c,h] = contour(reshape(x_gnd(:,1), x_shape), reshape(x_gnd(:,2), x_shape), reshape(y_predict, x_shape), ...
      linspace(-1, 3, 5));
   clabel(c,h);
   hold on;
   scatter(lse.collected_locs(:,1), lse.collected_locs(:,2), 50, 'k','s','filled');
   ax = gca;
   ax.YDir = 'normal';
   title('LSE choices and posterior');

   figure(5); clf; 
   imagesc(reshape(u, x_shape));
   ax = gca;
   ax.YDir = 'normal';
   title('LSE utility');

   pause(1);
end
