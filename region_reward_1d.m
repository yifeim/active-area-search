%%% draw prior

set(0,'defaultaxesposition',[0.1300    0.1100    0.7750    0.8150]) % 0.1300    0.1100    0.7750    0.8150
set(0,'defaultaxeslinewidth', .5); % 0.5
set(0,'defaultaxesfontsize', 20);


writegiftrue = true;

x = (-1:.005:1)';
% arm_set = round([.5:1:25.5]*length(x)/26);
arm_set = 1:length(x);

region_ends   = round([1:6]*length(x)/6);
region_starts = round([0:5]*length(x)/6)+1;
region_mids   = round((region_starts + region_ends)/2);
region_X   = cell2mat(arrayfun(@(n) { ...
  accumarray((region_starts(n):region_ends(n))', ...
  1/(region_ends(n) - region_starts(n)+1), [length(x), 1])}, ...
  1:length(region_starts)))';
region_set   = 1:length(region_starts);
region_width = sum(region_X'>0, 1);

alpha = norminv(1-.05/length(arm_set));
ylim = [0 5];
threshold = 3.2;

K0 = .9^2 * (range(ylim)/2)^2 / alpha^2 * ...
  exp(-squareform( pdist(x).^2 ) / .2.^2);
% K0 = K0 + diag(diag(K0) * 1e-5);
s_n = .1;

mu0 = mean(ylim) * ones(size(x));

rng shuffle
disp(rng)
rng(1065105080);
y = mvnrnd(mu0, K0 + s_n^2 * eye(size(K0)))';
FR = region_X * y;


which = @(x, v) find(abs(x-v)==min(abs(x-v)), 1);

while FR(which(x(region_mids),-.5)) < threshold || FR(which(x(region_mids),.5)) < threshold ...
    || FR(1) > threshold || FR(end) > threshold || FR(which(x(region_mids),0)) > threshold
  rng shuffle
  disp(rng)
  y = mvnrnd(mu0, K0 + s_n^2 * eye(size(K0)))';
  
  FR = region_X * y;
  
  figure(1); clf;
  plot(x,y);
  hold on
  plot([x(1) x(end)], [threshold threshold], '-r');
  plot(x(region_mids), FR, 'o');
  axis([-1 1 0 5]);
  drawnow;
  
  %   pause(.1);
end



mu = mu0;
K = K0;
sf = sqrt(diag(K));

muR = region_X * mu;
sfR = sqrt(diag(region_X * K * region_X'));
alphaR = norminv(1-.05/length(region_starts)); %.9
clR = muR - alphaR * sfR;
cuR = muR + alphaR * sfR;

pool_set = arm_set; 
pulled_ids = [];
xs = [];
ys = [];
region_pool  = 1:length(region_starts);

%%% sample
for sample_id = 1:20
  
  LK = region_X * K;
  
  uga = [];
  utility = [];
  for s = 1:length(x)
    sfR_update = abs(LK(:,s) * (K(s,s) + s_n.^2)^-.5);
    uga(:, s) = normcdf( ( muR - threshold - sqrt(sfR.^2 - sfR_update.^2)*alphaR ) ./ sfR_update );
  end
  ug = uga;
  ug(comple(region_pool, length(region_set)), :) = nan;
  ug(:, comple(pool_set, length(x))) = nan;
  utility = nansum(ug, 1);
  %   utility(:, comple(pool_set, length(x))) = nan;
  
  if max(utility)<1e-14
    break;
  end
  
  last_id = find(utility > max(utility) - 1e-10*range(utility));
  if length(last_id)>1
    last_id = randsample(last_id, 1);
  end
  
  if sample_id > 0
    pool_set(pool_set == last_id) = [];
    pulled_ids(end+1) = last_id;
  
    xs(end+1) = x(last_id);
    ys(end+1) = y(last_id);
  
    %%% update posterior
    [mu, K] = Gaussian_posterior('cov', mu, K, ...
      accumarray([1,last_id], 1, [1, length(x)]), ys(end), s_n);

    sf = sqrt(diag(K));
    muR = region_X * mu;
    sfR = sqrt(diag(region_X * K * region_X'));
    clR = muR - alphaR * sfR;
    cuR = muR + alphaR * sfR;
  
    %%% permanent persistence
    region_pool(clR(region_pool)>threshold) = [];
  end


  %%% draw result
  figure(1); clf;
  set(gcf,'position',[440   378   640   300],'color',[1 1 1]);

  plot([x(1) x(end)], [threshold threshold], '-r');
  hold on;

  Hs = shadedErrorBar(x,mu, alpha*sf, '--b', .5);
  Hs.edge(1).LineStyle = 'none';
  Hs.edge(2).LineStyle = 'none';
  %   Hs.mainLine.LineStyle = 'none';

  %   for re = region_ends
  %     line([x(re); x(re)], [-1.2*alpha*sf0(re), 1.2*alpha*sf0(re)]);
  %   end
  for rid = 1:length(region_mids)
    if ~any(region_pool == rid) % pass threshold
      shadedErrorBar(x(region_starts(rid):region_ends(rid)), ...
        repmat(muR(rid),[region_width(rid),1]), repmat(alphaR*sfR(rid),[region_width(rid),1]), 'g',.2);
    else %if threshold > cuR(rid) % below
      shadedErrorBar(x(region_starts(rid):region_ends(rid)), ...
        repmat(muR(rid),[region_width(rid),1]), repmat(alphaR*sfR(rid),[region_width(rid),1]), 'k',.2);
      %     else % partial
      %       shadedErrorBar(x(region_starts(rid):region_ends(rid)), ...
      %         repmat(muR(rid),[region_width(rid),1]), [repmat(alphaR*sfR(rid),[region_width(rid),1]), repmat(threshold-muR(rid),[region_width(rid),1])], 'k',.2);
      %
      %       shadedErrorBar(x(region_starts(rid):region_ends(rid)), ...
      %         repmat(muR(rid),[region_width(rid),1]), [-repmat(muR(rid)-threshold,[region_width(rid),1]), -repmat(alphaR*sfR(rid),[region_width(rid),1])], 'y',.2);
    end
  end
  %   for a=arm_set
  %     line([x(a); x(a)], [-1.2*alpha*sf0(a), mu(a) + alpha*sf(a)]);
  %   end
  for i=1:length(pulled_ids)
    plot(xs(i), ys(i), 'o', 'markersize', 30);
    % text(xs(i), ys(i) + 1, num2str(i), 'fontsize', 20, 'horizontalalignment', 'center');
  end

  xlabel('items')
  ylabel('value')
  
  axis([-1 1 0 5]); set(gca,'ytick',[0 1 2 3 4 5], 'xtick',[],'box','off');
  drawnow;
  
  if writegiftrue
    frame = getframe(1);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    if sample_id == 1
      imwrite(imind, cm, [mfilename, '.gif'], 'gif', 'Loopcount', 1, 'delaytime', 1);
    else
      imwrite(imind, cm, [mfilename, '.gif'], 'gif', 'WriteMode', 'append', 'delaytime', 1);
    end
  end
  
  figure(2); clf;
  set(gcf,'position',[440   378   640   300],'color',[1 1 1]);
  plot(x(arm_set), ug(:, arm_set)');
  set(legend('g_1','g_2','g_3','g_4','g_5','g_6'),'location','northwest','box','off');
  set(gca,'position',[0.1300    0.1100    0.7750    0.8000], 'xlim', [-1 1]);
  drawnow;
  
  if writegiftrue
    frame = getframe(2);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    if sample_id == 1
      imwrite(imind, cm, [mfilename, '_utility.gif'], 'gif', 'Loopcount', 1, 'delaytime', 1);
    else
      imwrite(imind, cm, [mfilename, '_utility.gif'], 'gif', 'WriteMode', 'append', 'delaytime', 1);
    end
  end

  pause(.1);
  
end
