%%% draw prior

set(0,'defaultaxesposition',[0.1300    0.1100    0.7750    0.8150]) % 0.1300    0.1100    0.7750    0.8150
set(0,'defaultaxeslinewidth', .5); % 0.5
set(0,'defaultaxesfontsize', 20);


writegiftrue = true;


x = (-1:.005:1)';
arm_set = round([.5:1:25.5]*length(x)/26);

region_ends = arm_set;
region_starts = arm_set;
region_mids   = round((region_starts + region_ends)/2);
region_X   = cell2mat(arrayfun(@(n) { ...
  accumarray((region_starts(n):region_ends(n))', ...
  1/(region_ends(n) - region_starts(n)+1), [length(x), 1])}, ...
  1:length(region_starts)))';
region_pool  = 1:length(region_starts);
region_set   = 1:length(region_starts);

alpha = norminv(1-.05/length(arm_set));
ylim = [0 5];
threshold = 3.5;

K0 = .9^2 * (range(ylim)/2)^2 / alpha^2 * ...
  exp(-squareform( pdist(x).^2 ) / .3.^2);
s_n = .1;

mu0 = mean(ylim) * ones(size(x));

rng shuffle
disp(rng)
rng(1054718343);
y = mvnrnd(mu0, K0 + s_n^2 * eye(size(K0)))';

while y(arm_set(5)) < threshold || y(arm_set(end-5)) < threshold ...
    || y(1) > threshold || y(end) > threshold || y(round((1+length(y))/2)) > threshold
  rng shuffle
  disp(rng)
  y = mvnrnd(mu0, K0 + s_n^2 * eye(size(K0)))';
  
  figure(1); clf;
  plot(x,y);
  hold on
  plot([x(1) x(end)], [threshold threshold], '-r');
  axis([-1 1 0 5]);
  drawnow;
  %   pause(.1);
end



mu = mu0;
K = K0;
sf = sqrt(diag(K));

muR = region_X * mu;
sfR = sqrt(diag(region_X * K * region_X'));
alphaR = alpha; %norminv(.9);
clR = muR - alphaR * sfR;
cuR = muR + alphaR * sfR;

pool_set = arm_set; 
pulled_ids = [];
xs = [];
ys = [];

%%% sample
for sample_id = 1:20

  %   ucb = mu + alpha * sf;
  %   ucb(comple(pool_set, length(ucb))) = nan;
  LK = region_X * K;

  ug = [];
  utility = [];
  for s = 1:length(x)
    sfR_update = abs(LK(:,s) * (K(s,s) + s_n.^2)^-.5);
    ug(:, s) = normcdf( ( muR - threshold - sqrt(sfR.^2 - sfR_update.^2)*alphaR ) ./ sfR_update );
  end
  ug(comple(region_pool, length(region_set)), :) = nan;
  ug(:, comple(pool_set, length(x))) = nan;
  
  %%% expected reward specific
  ug(ismember(arm_set, pulled_ids), :) = nan;
  utility = nansum(ug, 1);
  
  if max(utility)<1e-5
    break;
  end
  
  last_id = find(utility > max(utility) - 1e-10);
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
  
  %%% draw arms
  for a=arm_set
    Hs = shadedErrorBar(x(a-2:a+2), mu(a-2:a+2), alpha*sf(a-2:a+2), '-b', .5);
    [Hs.edge(:).LineStyle] = deal('none','none');
    Hs.mainLine.LineStyle = 'none';
  end
  
  %%% draw regions
  for rid = 1:length(region_mids)
    if ~any(region_pool == rid) % pass threshold
      shadedErrorBar(x(region_mids(rid)-4:region_mids(rid)+4), ...
        repmat(muR(rid),[9,1]), repmat(alphaR*sfR(rid),[9,1]), 'g');
    else %if threshold > cuR(rid) % below
      shadedErrorBar(x(region_mids(rid)-4:region_mids(rid)+4), ...
        repmat(muR(rid),[9,1]), repmat(alphaR*sfR(rid),[9,1]), 'k');
      %     else % partial
      %       shadedErrorBar(x(region_mids(rid)-4:region_mids(rid)+4), ...
      %         repmat(muR(rid),[9,1]), [repmat(alphaR*sfR(rid),[9,1]), repmat(threshold-muR(rid),[9,1])], 'k');
      %
      %       shadedErrorBar(x(region_mids(rid)-4:region_mids(rid)+4), ...
      %         repmat(muR(rid),[9,1]), [-repmat(muR(rid)-threshold,[9,1]), -repmat(alphaR*sfR(rid),[9,1])], 'k');
    end
  end
  
  for i=1:length(pulled_ids)
    plot(xs(i), ys(i), 'o', 'markersize', 30);
    %     text(xs(i), ys(i)+1, num2str(i), 'horizontalalignment','center');
    %     if ys(i) > threshold
    %       text(xs(i), ys(i), char(hex2dec('2713')), 'color','red','fontsize', 20, 'horizontalalignment', 'center','verticalalignment','middle');
    %     else
    %       plot(xs(i), ys(i), 'x', 'markersize', 15, 'color','black','linewidth',2);
    %     end
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

  pause(.1);
  
  if max(mu(pool_set) + alpha*sf(pool_set)) < threshold
    break;
  end
  
end

disp(sample_id)
